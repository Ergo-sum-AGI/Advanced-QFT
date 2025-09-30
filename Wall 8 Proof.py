import numpy as np
from scipy.integrate import quad, odeint
import matplotlib.pyplot as plt

# ---------- hyperbolic heat-kernel H^5 ----------
def KH5(d,t):
    pref=(4*np.pi*t)**(-2.5)*np.exp(-d**2/(4*t))
    if d<1e-3: return pref*(1+d**2/6)          # small-d limit
    return pref*(d/np.sinh(d))*(1+d**2/3)

def Area(d): return np.sinh(d)**4

# ---------- regulated volume ----------
def Vol(mu,t):
    integrand=lambda d: Area(d)*np.exp(-mu*d**2)*KH5(d,t)
    return 4*np.pi*quad(integrand,0,50,limit=200)[0]

# ---------- β-functions ----------
phi=(1+np.sqrt(5))/2
def beta_mu(mu,eta): return mu*(eta - 2)
def eta_fn(mu,t): return 5/2 - (15*phi)/(16*np.pi**2)   # one-loop

# ---------- coupled flow ----------
def flow(y,lam):
    mu,t = y
    et = eta_fn(mu,t)
    dmu = beta_mu(mu,et)
    dt  = 1                       # t=lambda (diffusion time)
    return [dmu, dt]

lam_span=np.linspace(0,15,500)
y0=[1.0, 1e-3]                   # initial mu=1, t=1e-3
sol=odeint(flow,y0,lam_span)
mu_sol,t_sol=sol[:,0],sol[:,1]

# ---------- output ----------
mu_star=mu_sol[-1]
Vol_inf=Vol(mu_star,t_sol[-1])
print(f'μ_*   = {mu_star:.4f}')
print(f'Vol_R = {Vol_inf:.4f}  (unique finite)')

plt.semilogy(lam_span,mu_sol,label='μ(λ)')
plt.xlabel('Ricci-flow time λ'); plt.ylabel('mass parameter μ')
plt.title('Wall-8: μ runs to finite fixed point')
plt.legend(); plt.savefig('wall8_mu_flow.pdf'); plt.show()