"""SmartMagDL finite-cylinder field and online smartphone observations.

SI internally (m, A/m, T). Public observations/normalization use microtesla.
See README.md for deliberate corrections to the MATLAB source.
"""
from dataclasses import dataclass, fields
from pathlib import Path
import json
import os
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from typing import Optional

MU0 = 4 * np.pi * 1e-7


@dataclass
class MagneticConfig:
    domain_m: float = 30.0
    earth_uT: tuple = (3.0, 29.0, 33.0)
    center_uT: float = float(np.linalg.norm([3, 29, 33]))
    scale_uT: float = 20.0
    noise_uT: float = 1.0
    offset_uT: float = 0.1
    tilt_deg: float = 2.0
    softiron_diag: float = 0.25
    softiron_offdiag: float = 0.05
    sensitivity_error: float = 0.001
    sensor_mode: str = 'calibrated'  # ideal, calibrated, raw
    shared_magnet_properties: bool = True
    # mat_file: str | None = None
    mat_file: Optional[str] = None

    gp_length_scale: float = 0.12
    gp_noise_uT: float = 1.0
    reconstructor: str = 'gp'
    # smartmag_checkpoint: str | None = None
    smartmag_checkpoint: Optional[str] = None
    # smartmag_scaling: str | None = None
    smartmag_scaling: Optional[str] = None
    smartmag_device: str = 'cpu'
    plot_grid_size: int = 300

    def __post_init__(self):
        if self.reconstructor not in {'gp', 'smartmagdl'}:
            raise ValueError('reconstructor must be gp or smartmagdl')
        if self.plot_grid_size < 2:
            raise ValueError('plot_grid_size must be at least 2')
        if self.domain_m <= 10 or self.scale_uT <= 0 or self.gp_length_scale <= 0:
            raise ValueError('domain_m must exceed 10 m; scale and GP length scale must be positive')
        if self.sensor_mode not in {'ideal', 'calibrated', 'raw'}:
            raise ValueError('sensor_mode must be ideal, calibrated, or raw')
        for key in ('noise_uT','offset_uT','tilt_deg','softiron_diag','softiron_offdiag',
                    'sensitivity_error','gp_noise_uT'):
            if getattr(self,key) < 0:
                raise ValueError(f'{key} cannot be negative')
        if np.shape(self.earth_uT) != (3,):
            raise ValueError('earth_uT must have three components')

    @classmethod
    def from_environment(cls):
        path = os.environ.get('SMARTMAG_CONFIG')
        if not path:
            return cls()
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown magnetic configuration keys: {sorted(unknown)}')
        for key in ('mat_file', 'smartmag_checkpoint', 'smartmag_scaling'):
            if data.get(key):
                data[key] = str((path.parent / data[key]).resolve())
        return cls(**data)


def cel(kc, p, c, s, tol=1e-10):
    """Vectorized positive-p Bulirsch CEL, as in ParallelBulirsch.m."""
    k, pp, cc, ss = [a.copy() for a in np.broadcast_arrays(
        np.abs(np.asarray(kc, float)), np.asarray(p, float),
        np.asarray(c, float), np.asarray(s, float))]
    if np.any(k <= 0) or np.any(pp <= 0):
        raise ValueError('CEL requires kc>0 and p>0; cylinder surface is singular')
    pp = np.sqrt(pp)
    ss = ss / pp
    em = np.ones_like(k)
    f = cc.copy(); cc = cc + ss / pp
    g = k / pp; ss = 2 * (ss + f * g); pp = pp + g
    g = em.copy(); em = em + k; kk = k.copy()
    for _ in range(100):
        if np.all(np.abs(g-k) <= g*tol):
            return (np.pi/2) * (ss + cc*em) / (em*(em+pp))
        k = 2*np.sqrt(kk); kk = k*em
        f = cc.copy(); cc = cc+ss/pp
        g = kk/pp; ss = 2*(ss+f*g); pp = pp+g
        g = em.copy(); em = em+k
    raise RuntimeError('Bulirsch iteration did not converge')


def cylinder_field(points_m, position_m, direction, diameter_m, length_m, magnetization):
    """Finite cylinder, arbitrary unit axial orientation; returns (N,3) T.

    Port of ParallelDerby/ParallelBfield. Uses the exact axial limit on-axis
    and integral fallback when rho equals the cylinder radius outside its ends.
    """
    pts = np.asarray(points_m, dtype=float).reshape(-1,3)
    axis = np.asarray(direction, dtype=float)
    if axis.shape != (3,) or np.linalg.norm(axis) == 0 or min(diameter_m,length_m) <= 0:
        raise ValueError('Invalid cylinder geometry')
    axis = axis / np.linalg.norm(axis)
    x = pts - np.asarray(position_m)
    z = x @ axis
    radial = x - z[:,None]*axis
    rho = np.linalg.norm(radial, axis=1)
    R = diameter_m/2
    zp, zm = z+length_m/2, z-length_m/2
    if np.any(np.isclose(rho,R,rtol=0,atol=1e-13) & (np.abs(z)<=length_m/2)):
        raise ValueError('Field on the magnet surface is undefined in this implementation')
    ap = R/np.sqrt(zp*zp+(rho+R)**2)
    am = R/np.sqrt(zm*zm+(rho+R)**2)
    bp = zp/np.sqrt(zp*zp+(rho+R)**2)
    bm = zm/np.sqrt(zm*zm+(rho+R)**2)
    kp = np.sqrt((zp*zp+(rho-R)**2)/(zp*zp+(rho+R)**2))
    km = np.sqrt((zm*zm+(rho-R)**2)/(zm*zm+(rho+R)**2))
    gamma = (R-rho)/(R+rho)
    b0 = MU0*magnetization/np.pi
    br = b0*(ap*cel(kp,1,1,-1)-am*cel(km,1,1,-1))
    regular = np.abs(gamma)>1e-12
    bz = np.empty_like(z)
    bz[regular] = b0*R/(R+rho[regular])*(
        bp[regular]*cel(kp[regular],gamma[regular]**2,1,gamma[regular])-
        bm[regular]*cel(km[regular],gamma[regular]**2,1,gamma[regular]))
    for i in np.flatnonzero(~regular):
        def integral(k):
            return quad(lambda t: 1/np.sqrt(np.cos(t)**2+k*k*np.sin(t)**2),
                        0,np.pi/2,epsabs=1e-11)[0]
        bz[i] = b0/2*(bp[i]*integral(kp[i])-bm[i]*integral(km[i]))
    unit_r = np.divide(radial,rho[:,None],out=np.zeros_like(radial),where=rho[:,None]>0)
    return br[:,None]*unit_r + bz[:,None]*axis


class MagneticScene:
    """One scene per episode; fixed calibration errors, fresh sample noise."""
    def __init__(self, seed=None, config=None):
        self.config = config or MagneticConfig.from_environment()
        self.rng = np.random.default_rng(seed)
        c, rng = self.config, self.rng
        self.interpolator = None
        if c.mat_file:
            self._load_mat(c.mat_file)
        else:
            n = int(rng.choice([1,5]))
            self.positions = np.column_stack((rng.uniform(-c.domain_m/2+5,c.domain_m/2-5,(n,2)), -np.ones(n)))
            self.directions = np.array([[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]])[rng.integers(0,4,n)]
            self.diameters = rng.uniform(.025,.030,n)
            self.lengths = rng.uniform(.8,1.2,n)
            self.magnetizations = (1.2706/MU0)/rng.uniform(1,2,n)
            if c.shared_magnet_properties:
                # MATLAB passes D(1), L(1), M(1) to ALL magnets.
                self.diameters[:] = self.diameters[0]
                self.lengths[:] = self.lengths[0]
                self.magnetizations[:] = self.magnetizations[0]
        self.S = np.diag(1-rng.uniform(0,c.sensitivity_error,3))
        self.T = np.diag(rng.uniform(1-c.softiron_diag,1+c.softiron_diag,3))
        self.T[~np.eye(3,dtype=bool)] = rng.uniform(-c.softiron_offdiag,c.softiron_offdiag,6)
        self.offset = rng.normal(0,c.offset_uT,3)
        self.observation_log = []

    def _load_mat(self,path):
        data = loadmat(path)
        required = {'xg','yg','Bxt_e','Byt_e','Bzt_e'}
        if not required <= data.keys():
            raise ValueError(f'Use a SmartMagDL TheorB MAT file containing {sorted(required)}')
        x, y = data['xg'].ravel()/1000, data['yg'].ravel()/1000
        B = np.stack([data[k] for k in ('Bxt_e','Byt_e','Bzt_e')],axis=-1)
        if B.shape != (len(y),len(x),3) or not np.all(np.isfinite(B)):
            raise ValueError('Invalid MAT field shape/values: expected [y,x,3] in tesla')
        if np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0):
            raise ValueError('MAT axes must be strictly increasing')
        d = self.config.domain_m/2
        if x[0]>-d or x[-1]<d or y[0]>-d or y[-1]<d:
            raise ValueError('MAT axes must cover the configured square domain')
        self.interpolator = RegularGridInterpolator((y,x),B,bounds_error=True)

    def world_field(self, xy):
        xy = np.asarray(xy,float).reshape(-1,2)
        if not np.all(np.isfinite(xy)) or np.any(xy < -1e-12) or np.any(xy > 1+1e-12):
            raise ValueError('Planner coordinates must be finite and in [0,1]')
        pos = (np.clip(xy,0,1)-.5)*self.config.domain_m
        if self.interpolator is not None:
            return self.interpolator(pos[:,[1,0]])*1e6
        pts = np.column_stack((pos,np.zeros(len(pos))))
        B = np.zeros_like(pts)
        for p,u,d,l,m in zip(self.positions,self.directions,self.diameters,self.lengths,self.magnetizations):
            B += cylinder_field(pts,p,u,d,l,m)
        return B*1e6 + np.asarray(self.config.earth_uT)

    def distribution_function(self, xy):
        """Noise-free normalized truth, used ONLY for offline evaluation."""
        return (np.linalg.norm(self.world_field(xy),axis=1)-self.config.center_uT)/self.config.scale_uT

    @staticmethod
    def rotation(yaw,pitch,roll):
        ca,sa,cb,sb,cr,sr = np.cos(yaw),np.sin(yaw),np.cos(pitch),np.sin(pitch),np.cos(roll),np.sin(roll)
        # Active Rz(yaw) Ry(pitch) Rx(roll), matching MATLAB's convention.
        return np.array([[ca*cb,ca*sb*sr-sa*cr,ca*sb*cr+sa*sr],
                         [sa*cb,sa*sb*sr+ca*cr,sa*sb*cr-ca*sr],[-sb,cb*sr,cb*cr]])

    def observe(self, xy, heading=0.):
        xy = np.asarray(xy,float).reshape(-1,2)
        headings = np.broadcast_to(heading,(len(xy),))
        values=[]
        for pt,b,h in zip(xy,self.world_field(xy),headings):
            if self.config.sensor_mode == 'ideal':
                measured = b
            else:
                pitch,roll = self.rng.normal(0,np.deg2rad(self.config.tilt_deg),2)
                body = self.rotation(h,pitch,roll) @ b
                raw = self.S @ self.T @ body + self.offset + self.rng.normal(0,self.config.noise_uT,3)
                # Known simulated calibration: no access to map truth needed.
                measured = np.linalg.solve(self.S @ self.T, raw-self.offset) if self.config.sensor_mode=='calibrated' else raw
            mag = float(np.linalg.norm(measured))
            values.append((mag-self.config.center_uT)/self.config.scale_uT)
            self.observation_log.append([*pt,float(h),mag])
        return np.asarray(values)
