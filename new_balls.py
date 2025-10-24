#!/usr/bin/env python3
"""
Ballistics Calculator — Tkinter GUI
- No SciPy required (pure-Python RK4); only numpy + stdlib + matplotlib for plots
- Simple 2D model: gravity + quadratic drag scaled by BC (G1-style proxy)
- Adds two live plots on the right: trajectory (top) and a crosshair-style reticle hold (bottom)
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# NEW: matplotlib for embedded plots
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

        # Make room for a graph column on the right
        for i in range(4):  # was 3
            self.columnconfigure(i, weight=(0 if i < 3 else 1))
        self.rowconfigure(99, weight=1)

        self._build_widgets()
        self._build_plots()

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

    # NEW: build plot area on the right
    def _build_plots(self):
        self.plot_frame = ttk.Frame(self)
        # Span many rows, sit in column 3 (right-hand side)
        self.plot_frame.grid(row=0, column=3, rowspan=100, sticky="nsew", padx=(12,0))
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.rowconfigure(1, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        # Top figure: trajectory
        self.fig_traj = Figure(figsize=(5.4, 2.6), dpi=100)
        self.ax_traj = self.fig_traj.add_subplot(111)
        self.ax_traj.set_title("Trajectory (relative to line of sight)")
        self.ax_traj.set_xlabel("Distance (yd)")
        self.ax_traj.set_ylabel("Height (in)")
        self.ax_traj.grid(True, alpha=0.3)
        self.canvas_traj = FigureCanvasTkAgg(self.fig_traj, master=self.plot_frame)
        self.canvas_traj.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Bottom figure: crosshair-style reticle (mils)
        self.fig_ret = Figure(figsize=(5.4, 2.6), dpi=100)
        self.ax_ret = self.fig_ret.add_subplot(111)
        self._init_reticle_axes()
        self.canvas_ret = FigureCanvasTkAgg(self.fig_ret, master=self.plot_frame)
        self.canvas_ret.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # Placeholders for dynamic artists
        self.traj_line = None
        self.zero_marker = None
        self.target_marker = None
        self.hold_dot = None

    def _init_reticle_axes(self):
        ax = self.ax_ret
        ax.clear()
        ax.set_title("Reticle hold (mils)")
        # Reticle viewbox (mils). Positive up by default; we invert y so holds are "down".
        x_lim = 6
        y_up, y_down = 4, 12
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(y_down, -y_up)  # invert so down is positive on screen (like a scope)
        ax.set_aspect('equal', adjustable='box')

        # Main crosshair lines
        ax.axhline(0, linewidth=1.2)
        ax.axvline(0, linewidth=1.2)

        # Tick hash marks each 1 mil (longer every 2)
        for x in range(-x_lim, x_lim+1):
            if x == 0: continue
            ax.plot([x, x], [-0.3 if x % 2 else -0.5, 0.3 if x % 2 else 0.5], linewidth=1)

        for y in range(1, y_down+1):  # below center
            ax.plot([-0.3 if y % 2 else -0.5, 0.3 if y % 2 else 0.5], [y, y], linewidth=1)
        for y in range(1, y_up+1):    # above center
            ax.plot([-0.3 if y % 2 else -0.5, 0.3 if y % 2 else 0.5], [-y, -y], linewidth=1)

        # Labels for quick read every 2 mils
        for y in range(2, y_down+1, 2):
            ax.text(0.65, y+0.15, f"{y} mil", fontsize=8)
        ax.set_xlabel("Wind hold (mils)")
        ax.set_ylabel("Elevation (mils)")

        # Light grid
        ax.grid(True, alpha=0.15)

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

        # Solve for bore angle and simulate (same as before)
        bore_angle = find_bore_angle(muzzle_v, bc, mass_kg, zero_m, wind_mps, dt)
        ts, xs, ys, vxs, vys = simulate_rk4(muzzle_v, bc, mass_kg, bore_angle, wind_mps, dt=dt, max_x=target_m*1.5)

        if target_m > xs[-1]:
            self._write_result("Target distance beyond simulation range.")
            return

        # Interpolate state at target range
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

        # Text results (unchanged content)
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

        # ---------- UPDATE PLOTS ----------
        self._update_trajectory_plot(xs, ys, zero_m, target_m)
        self._update_reticle_plot(hold_mil)

    # NEW: draw top plot (convert to yards/inches, clamp to target)
    def _update_trajectory_plot(self, xs_m, ys_m, zero_m, target_m):
        ax = self.ax_traj
        ax.clear()
        ax.set_title("Trajectory (relative to line of sight)")
        x_yd = xs_m / 0.9144
        y_in = ys_m / 0.0254

        # Keep only up to target distance for the visible curve
        mask = x_yd <= (target_m / 0.9144)
        ax.plot(x_yd[mask], y_in[mask], linewidth=1.8)

        # Zero line and markers
        ax.axhline(0, color='k', linewidth=1, alpha=0.4)
        ax.axvline(zero_m/0.9144, color='k', linestyle='--', linewidth=1, alpha=0.3)
        ax.text(zero_m/0.9144, 0, "  zero", va='bottom', fontsize=8)

        # Target marker
        target_yd = target_m / 0.9144
        # Interpolate y at target for marker
        y_target_in = float(np.interp(target_m, xs_m, ys_m)) / 0.0254
        ax.plot([target_yd], [y_target_in], marker='o')
        ax.text(target_yd, y_target_in, f"  target ({target_yd:.0f} yd)", va='bottom', fontsize=8)

        ax.set_xlim(0, target_yd)
        # Nice y range padding
        ymin = min(0, y_in.min()) - 2
        ymax = max(0, y_in.max()) + 2
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel("Distance (yd)")
        ax.set_ylabel("Height (in)")
        ax.grid(True, alpha=0.3)
        self.canvas_traj.draw_idle()

    # NEW: draw bottom reticle plot (vertical hold in mils; wind = 0)
    def _update_reticle_plot(self, hold_mil):
        self._init_reticle_axes()  # rebuild grid/crosshair each time
        ax = self.ax_ret

        # Place a dot at the required elevation hold, x=0 (no wind in 2D model)
        # Positive "down" on screen; hold_mil is positive (down) when dropping.
        ax.plot([0], [hold_mil], marker='o', markersize=6)
        ax.text(0.3, hold_mil + 0.35, f"{hold_mil:.1f} mil", fontsize=9)

        # Small helper text
        ax.text(-5.8, -3.2, "Center = POA\nDot = required hold\n(Vertical only; wind not modeled in 2D)", fontsize=7)

        self.canvas_ret.draw_idle()

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
    # Expand main window layout nicely
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    app = BallisticsApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

