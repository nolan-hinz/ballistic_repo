#!/usr/bin/env python3
"""
Ballistics Calculator — Tkinter GUI
- No SciPy required (pure-Python RK4); only numpy + stdlib
- Simple 2D model: gravity + quadratic drag scaled by BC (G1-style proxy)
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

g = 9.80665
k_default = 0.00055
MAX_SIM_TIME = 30.0
DEFAULT_DT = 0.003

def fps_to_mps(v_fps): return v_fps * 0.3048
def yd_to_m(y):       return y * 0.9144
def inches_to_m(i):   return i * 0.0254
def grains_to_kg(gr): return gr * 0.00006479891

def simulate_rk4(muzzle_v, bc, mass, bore_angle_rad, wind_x, dt=DEFAULT_DT, max_x=20000.0):
    vx = muzzle_v * math.cos(bore_angle_rad)
    vy = muzzle_v * math.sin(bore_angle_rad)
    x = 0.0; y = 0.0; t = 0.0
    ts, xs, ys, vxs, vys = [t], [x], [y], [vx], [vy]

    def deriv(vx, vy):
        v_rel_x = vx - wind_x
        v_rel_y = vy
        v_rel = math.hypot(v_rel_x, v_rel_y)
        if v_rel <= 1e-12:
            a_dx = 0.0; a_dy = 0.0
        else:
            a_drag_mag = (k_default / max(bc, 1e-6)) * v_rel * v_rel
            a_dx = -a_drag_mag * (v_rel_x / v_rel)
            a_dy = -a_drag_mag * (v_rel_y / v_rel)
        ax = a_dx
        ay = a_dy - g
        return ax, ay

    while t < MAX_SIM_TIME and x < max_x and y > -500.0:
        ax1, ay1 = deriv(vx, vy)
        k1_vx, k1_vy = ax1*dt, ay1*dt
        k1_x,  k1_y  = vx*dt, vy*dt

        ax2, ay2 = deriv(vx + 0.5*k1_vx, vy + 0.5*k1_vy)
        k2_vx, k2_vy = ax2*dt, ay2*dt
        k2_x,  k2_y  = (vx + 0.5*k1_vx)*dt, (vy + 0.5*k1_vy)*dt

        ax3, ay3 = deriv(vx + 0.5*k2_vx, vy + 0.5*k2_vy)
        k3_vx, k3_vy = ax3*dt, ay3*dt
        k3_x,  k3_y  = (vx + 0.5*k2_vx)*dt, (vy + 0.5*k2_vy)*dt

        ax4, ay4 = deriv(vx + k3_vx, vy + k3_vy)
        k4_vx, k4_vy = ax4*dt, ay4*dt
        k4_x,  k4_y  = (vx + k3_vx)*dt, (vy + k3_vy)*dt

        x  += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6.0
        y  += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6.0
        vx += (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) / 6.0
        vy += (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) / 6.0

        t  += dt
        ts.append(t); xs.append(x); ys.append(y); vxs.append(vx); vys.append(vy)
        if math.hypot(vx, vy) < 5.0:
            break

    return np.array(ts), np.array(xs), np.array(ys), np.array(vxs), np.array(vys)

def height_error_at_zero(angle, muzzle_v, bc, mass_kg, zero_m, wind_mps, dt):
    _, xs, ys, _, _ = simulate_rk4(muzzle_v, bc, mass_kg, angle, wind_mps, dt=dt, max_x=max(10.0, zero_m*1.2))
    if zero_m <= xs[0]:
        y_at = ys[0]
    elif zero_m >= xs[-1]:
        y_at = ys[-1]
    else:
        y_at = float(np.interp(zero_m, xs, ys))
    return y_at

def find_bore_angle(muzzle_v, bc, mass_kg, zero_m, wind_mps, dt):
    a, b = -0.12, 0.12
    fa = height_error_at_zero(a, muzzle_v, bc, mass_kg, zero_m, wind_mps, dt)
    fb = height_error_at_zero(b, muzzle_v, bc, mass_kg, zero_m, wind_mps, dt)
    if fa == 0: return a
    if fb == 0: return b
    if fa*fb > 0:
        for ang in np.linspace(-0.3, 0.3, 25):
            f = height_error_at_zero(ang, muzzle_v, bc, mass_kg, zero_m, wind_mps, dt)
            if f == 0: return ang
            if f*fa < 0:
                b, fb = ang, f
                break
        else:
            return 0.0
    for _ in range(28):
        m = 0.5*(a+b)
        fm = height_error_at_zero(m, muzzle_v, bc, mass_kg, zero_m, wind_mps, dt)
        if abs(fm) < 1e-3:
            return m
        if fa*fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5*(a+b)

class BallisticsApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=12)
        self.master.title("Ballistics Calculator")
        self.grid(sticky="nsew")
        self._build_widgets()
        for i in range(3): self.columnconfigure(i, weight=1)
        self.rowconfigure(99, weight=1)

    def _build_widgets(self):
        r = 0
        ttk.Label(self, text="Muzzle velocity").grid(row=r, column=0, sticky="w")
        self.mv_var = tk.StringVar(value="2700")
        ttk.Entry(self, textvariable=self.mv_var, width=10).grid(row=r, column=1, sticky="w")
        self.mv_unit = ttk.Combobox(self, values=["fps","m/s"], width=6, state="readonly")
        self.mv_unit.set("fps"); self.mv_unit.grid(row=r, column=2, sticky="w")

        r+=1
        ttk.Label(self, text="Ballistic Coefficient (G1)").grid(row=r, column=0, sticky="w")
        self.bc_var = tk.StringVar(value="0.45")
        ttk.Entry(self, textvariable=self.bc_var, width=10).grid(row=r, column=1, sticky="w")

        r+=1
        ttk.Label(self, text="Bullet mass").grid(row=r, column=0, sticky="w")
        self.mass_var = tk.StringVar(value="150")
        ttk.Entry(self, textvariable=self.mass_var, width=10).grid(row=r, column=1, sticky="w")
        self.mass_unit = ttk.Combobox(self, values=["grains"], width=6, state="readonly")
        self.mass_unit.set("grains"); self.mass_unit.grid(row=r, column=2, sticky="w")

        r+=1
        ttk.Label(self, text="Zero distance").grid(row=r, column=0, sticky="w")
        self.zero_var = tk.StringVar(value="100")
        ttk.Entry(self, textvariable=self.zero_var, width=10).grid(row=r, column=1, sticky="w")
        self.zero_unit = ttk.Combobox(self, values=["yd","m"], width=6, state="readonly")
        self.zero_unit.set("yd"); self.zero_unit.grid(row=r, column=2, sticky="w")

        r+=1
        ttk.Label(self, text="Target distance").grid(row=r, column=0, sticky="w")
        self.target_var = tk.StringVar(value="300")
        ttk.Entry(self, textvariable=self.target_var, width=10).grid(row=r, column=1, sticky="w")
        self.target_unit = ttk.Combobox(self, values=["yd","m"], width=6, state="readonly")
        self.target_unit.set("yd"); self.target_unit.grid(row=r, column=2, sticky="w")

        r+=1
        ttk.Label(self, text="Sight height above bore").grid(row=r, column=0, sticky="w")
        self.sight_var = tk.StringVar(value="1.5")
        ttk.Entry(self, textvariable=self.sight_var, width=10).grid(row=r, column=1, sticky="w")
        self.sight_unit = ttk.Combobox(self, values=["in","mm"], width=6, state="readonly")
        self.sight_unit.set("in"); self.sight_unit.grid(row=r, column=2, sticky="w")

        r+=1
        ttk.Label(self, text="Crosswind speed").grid(row=r, column=0, sticky="w")
        self.wind_var = tk.StringVar(value="0")
        ttk.Entry(self, textvariable=self.wind_var, width=10).grid(row=r, column=1, sticky="w")
        self.wind_unit = ttk.Combobox(self, values=["mph","m/s"], width=6, state="readonly")
        self.wind_unit.set("mph"); self.wind_unit.grid(row=r, column=2, sticky="w")

        r+=1
        ttk.Label(self, text="Time step dt (s)").grid(row=r, column=0, sticky="w")
        self.dt_var = tk.StringVar(value=str(DEFAULT_DT))
        ttk.Entry(self, textvariable=self.dt_var, width=10).grid(row=r, column=1, sticky="w")

        r+=1
        ttk.Button(self, text="Calculate", command=self.calculate).grid(row=r, column=0, columnspan=3, sticky="ew", pady=(6,8))

        r+=1
        ttk.Separator(self, orient="horizontal").grid(row=r, column=0, columnspan=3, sticky="ew", pady=6)

        r+=1
        self.result_text = tk.Text(self, height=14, width=60, wrap="word")
        self.result_text.grid(row=r, column=0, columnspan=3, sticky="nsew")
        self.result_text.configure(state="disabled")

        r+=1
        ttk.Label(self, foreground="#666",
                  text="Note: simplified BC-based drag. For long-range precision, use G1/G7 tables.")\
            .grid(row=r, column=0, columnspan=3, sticky="w", pady=(6,0))

    def _vel_to_mps(self, val, unit):
        v = float(val);  return fps_to_mps(v) if unit == "fps" else v
    def _dist_to_m(self, val, unit):
        d = float(val);  return yd_to_m(d) if unit == "yd" else d
    def _sight_to_m(self, val, unit):
        s = float(val);  return inches_to_m(s) if unit == "in" else (s/1000.0 if unit=="mm" else s)
    def _wind_to_mps(self, val, unit):
        w = float(val);  return w*0.44704 if unit == "mph" else w

    def calculate(self):
        try:
            muzzle_v = self._vel_to_mps(self.mv_var.get(), self.mv_unit.get())
            bc       = float(self.bc_var.get())
            mass_kg  = grains_to_kg(float(self.mass_var.get()))
            zero_m   = self._dist_to_m(self.zero_var.get(), self.zero_unit.get())
            target_m = self._dist_to_m(self.target_var.get(), self.target_unit.get())
            sight_m  = self._sight_to_m(self.sight_var.get(), self.sight_unit.get())
            wind_mps = self._wind_to_mps(self.wind_var.get(), self.wind_unit.get())
            dt       = float(self.dt_var.get())
        except Exception as e:
            messagebox.showerror("Input error", f"Please check your inputs.\n\n{e}")
            return
        if min(muzzle_v, bc, mass_kg, zero_m, target_m, dt) <= 0:
            messagebox.showerror("Input error", "All numeric inputs must be positive.")
            return

        bore_angle = find_bore_angle(muzzle_v, bc, mass_kg, zero_m, wind_mps, dt)
        ts, xs, ys, vxs, vys = simulate_rk4(muzzle_v, bc, mass_kg, bore_angle, wind_mps, dt=dt, max_x=target_m*1.5)

        if target_m > xs[-1]:
            self._write_result("Target distance beyond simulation range."); return

        y_t   = float(np.interp(target_m, xs, ys))
        vx_t  = float(np.interp(target_m, xs, vxs))
        vy_t  = float(np.interp(target_m, xs, vys))
        t_t   = float(np.interp(target_m, xs, ts))

        speed = math.hypot(vx_t, vy_t)
        energy_j = 0.5 * mass_kg * speed * speed

        drop_m  = -y_t
        drop_in = drop_m / 0.0254
        target_yd = target_m / 0.9144
        moa_per_in_at_range = (target_yd / 100.0) * 1.047
        hold_moa = drop_in / moa_per_in_at_range if moa_per_in_at_range > 0 else float("nan")
        hold_mil = drop_m / (target_m * 0.001) if target_m > 0 else float("nan")

        txt = []
        txt.append("=== Results ===")
        txt.append(f"Bore angle: {bore_angle*180.0/math.pi:.3f}°  (zero @ {float(self.zero_var.get()):.0f} {self.zero_unit.get()})")
        txt.append("")
        txt.append(f"At target: {float(self.target_var.get()):.0f} {self.target_unit.get()}  ({target_m:.1f} m)")
        txt.append(f"  Time of flight: {t_t:.3f} s")
        txt.append(f"  Vertical offset y (+ above LOS): {y_t:.4f} m")
        txt.append(f"  Drop (below LOS): {drop_in:.2f} in  ({drop_m*100:.1f} cm)")
        txt.append(f"  Holdover: {hold_moa:.2f} MOA   |   {hold_mil:.2f} mil")
        txt.append(f"  Remaining speed: {speed:.1f} m/s  ({speed/0.3048:.1f} ft/s)")
        txt.append(f"  Kinetic energy:  {energy_j:.1f} J")
        self._write_result("\n".join(txt))

    def _write_result(self, s: str):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", s)
        self.result_text.configure(state="disabled")

def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except tk.TclError:
        pass
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    app = BallisticsApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()



