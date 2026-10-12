"""Convex local-statement audits; no model fitting and no held-out selection.

Uses SciPy's HiGHS LP interface and public SLSQP for a whitened positive-definite QP. All primal/dual
claims are checked from arrays, not from a success string. Floating-point
bounds are not directed-rounding proofs.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular, null_space
from scipy.optimize import linprog, OptimizeWarning, minimize, LinearConstraint, nnls
import warnings
warnings.filterwarnings('ignore', category=OptimizeWarning, message='Unrecognized options detected:.*threads.*')
HIGHS_BACKEND = 'scipy.optimize.linprog/HiGHS LP; whitened SLSQP QP with NNLS dual recovery'


def pointwise_bound(P: np.ndarray, j: int, eps: float = .1) -> dict:
    """Maximize d[j] under abs(P d)<=eps, with primal and dual evidence."""
    n,q=P.shape
    c=np.eye(q)[j]
    B=np.vstack([P,-P]); ub=np.full(2*n,eps)
    result=linprog(-c,A_ub=B,b_ub=ub,bounds=[(None,None)]*q,
                   method='highs',options={'dual_feasibility_tolerance':1e-9,
                                           'primal_feasibility_tolerance':1e-9,
                                           'threads':1})
    if not result.success:
        return {'status':str(result.status),'message':result.message}
    d=np.asarray(result.x); raw=-np.asarray(result.ineqlin.marginals)
    mu=np.maximum(raw,0.)
    residual=B.T@mu-c
    smin=np.linalg.svd(P,compute_uv=False)[-1]
    # Feasible lower witness via symmetric radial contraction, not bound relaxation.
    factor=min(1.,eps/max(float(np.max(np.abs(P@d))),np.finfo(float).tiny))
    d=d*factor
    lower=float(d[j])
    upper=float(eps*(mu.sum()+np.sqrt(n)*np.linalg.norm(residual)/smin))
    return {'status':'optimal','d':d,'mu':mu,'raw_mu':raw,
            'lower':lower,'upper':upper,'residual':float(np.linalg.norm(residual)),
            'max_training':float(np.max(np.abs(P@d))),
            'gap':upper-lower,'radial_factor':factor}


def minimum_cost(P:np.ndarray,H:np.ndarray,J:float,j:int,target:float,eps:float=.1,initial:np.ndarray|None=None) -> dict:
    """Sharp minimum-cost statement change with constructive primal/dual bounds.

    With H=C C^T, d=C^{-T} x makes the objective ||x||^2. The QP is
    min .5||x||^2 subject to |P C^{-T}x|<=eps and (C^{-T}x)[j]=target.
    The returned objective is d^T H d/J (twice SLSQP's objective / J).
    A solver-only inward margin creates feasible upper witnesses. Multipliers
    are reconstructed independently by nonnegative least squares on active
    inequalities, with an unrestricted equality multiplier. Dual evaluation
    uses ORIGINAL epsilon, so gives a lower bound for the original problem.
    """
    import time
    start=time.time();n,q=P.shape
    C=cholesky(H,lower=True);M=solve_triangular(C.T,np.eye(q),lower=False)
    B=P@M;v=M[j];eps_solve=eps*(1-1e-7)
    if initial is None:
        # No target/test data used to initialize or solve this perturbation.
        lp=pointwise_bound(P,j,eps)
        if lp['status']!='optimal' or lp['lower']<=abs(target):
            return {'status':'initial_feasibility_unresolved'}
        initial=lp['d']*(target/lp['lower'])
    xinit=C.T@initial
    opt=minimize(lambda x:.5*x@x,xinit,jac=lambda x:x,method='SLSQP',
       constraints=[LinearConstraint(v[None,:],target,target),
                    LinearConstraint(B,-eps_solve,eps_solve)],
       options={'maxiter':100,'ftol':1e-12,'disp':False})
    x=opt.x;d=M@x;pr=P@d
    active=np.flatnonzero(eps_solve-np.abs(pr)<1e-7*max(1.,eps))
    Bactive=np.sign(pr[active])[:,None]*B[active]
    N=null_space(v[None,:])
    if len(active):
        mu,_=nnls(N.T@Bactive.T,-2*N.T@x,maxiter=10000)
    else: mu=np.zeros(0)
    nu=float(-(v@(2*x+Bactive.T@mu))/(v@v))
    upper_mu=np.zeros(n);lower_mu=np.zeros(n)
    upper_mu[active[pr[active]>0]]=mu[pr[active]>0]
    lower_mu[active[pr[active]<0]]=mu[pr[active]<0]
    qlin=P.T@(upper_mu-lower_mu);qlin[j]+=nu
    dual=-.25*qlin@cho_solve((C,True),qlin)-eps*mu.sum()-nu*target
    cost=float(d@H@d)
    return {'status':'optimal' if opt.success else 'SLSQP_'+str(opt.status),
       'solver_message':opt.message,'d':d,'upper_mu':upper_mu,'lower_mu':lower_mu,'nu':nu,
       'cost':cost/J,'dual_lower':float(dual/J),'gap':float((cost-dual)/J),
       'max_training':float(np.max(np.abs(pr))),'target_error':float(abs(d[j]-target)),
       'stationarity_inf':float(np.max(np.abs(2*H@d+qlin))),
       'iterations':int(opt.nit),'time':time.time()-start,'raw_solver_cost':float(2*opt.fun)}
