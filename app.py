import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration & Dark Mode Styling ---
st.set_page_config(
    page_title="Turbofan Cycle Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force Dark Mode Styling across Streamlit elements and Matplotlib defaults
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "#0b0f19",
    "axes.facecolor": "#0f172a",
    "savefig.facecolor": "#0b0f19",
    "text.color": "#f8fafc",
    "axes.labelcolor": "#38bdf8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "axes.edgecolor": "#334155"
})

st.markdown("""
    <style>
        /* Global Dark Theme Backgrounds */
        .stApp {
            background-color: #0b0f19;
            color: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            color: #f8fafc;
            border-right: 1px solid #1e293b;
        }
        [data-testid="stSidebar"] h3 {
            color: #38bdf8 !important;
            font-size: 16px;
            font-weight: 600;
            border-bottom: 1px solid #334155;
            padding-bottom: 6px;
        }
        [data-testid="stSidebar"] label {
            color: #cbd5e1 !important;
            font-size: 13px;
        }
        .stNumberInput input {
            background-color: #1e293b;
            color: #38bdf8;
            border: 1px solid #475569;
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 13px;
        }
        /* Table and Code Block Styling for Dark Mode */
        .stTable {
            background-color: #0f172a;
            color: #f8fafc;
        }
        code {
            color: #38bdf8 !important;
            background-color: #1e293b !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Core Cycle Function (Cached for Performance) ---
@st.cache_data
def run_cycle(Alt_ft, Tt5, pi_f, pi_lpc, pi_hpc, BPR, M_a, m_dot_core_sl, LHV, eta_d, eta_f, eta_lpc, eta_hpc, eta_b, pi_b, eta_hpt, eta_lpt, eta_mhp, eta_mlp, eta_fn, eta_cn, gamma_a, R_a, cp_a, gamma_g, R_g, cp_g):
    out = {}
    out['F_net'] = float('nan')
    alt_m = Alt_ft * 0.3048
    
    # Standard Atmosphere Model
    if alt_m <= 11000:
        Ta = 288.15 - 0.0065 * alt_m
        Pa = 101325 * (Ta / 288.15)**5.25588
    else:
        Ta = 216.65
        Pa = 22632 * np.exp(-9.80665 * 0.0289644 * (alt_m - 11000) / (8.31432 * 216.65))
    
    rho_0 = 101325 / (287 * 288.15)
    rho_a = Pa / (R_a * Ta)
    sigma = rho_a / rho_0
    
    m_dot_core = m_dot_core_sl * sigma
    out['m_dot_core'] = m_dot_core
    
    aa = np.sqrt(gamma_a * R_a * Ta)
    Va = M_a * aa
    Tta = Ta * (1 + (gamma_a - 1)/2 * M_a**2)
    Pta = Pa * (1 + (gamma_a - 1)/2 * M_a**2)**(gamma_a / (gamma_a - 1))
    
    Pt2 = Pta * eta_d
    Tt2 = Tta
    m_dot_fan = BPR * m_dot_core
    m_dot_total = m_dot_core + m_dot_fan
    
    Pt10 = Pt2 * pi_f 
    Tt10 = Tt2 + (Tt2 * (pi_f)**((gamma_a - 1)/gamma_a) - Tt2) / eta_f
    
    Pt3  = Pt10 * pi_lpc 
    Tt3  = Tt10 + (Tt10 * (pi_lpc)**((gamma_a - 1)/gamma_a) - Tt10) / eta_lpc
    
    Pt4  = Pt3 * pi_hpc  
    Tt4  = Tt3 + (Tt3 * (pi_hpc)**((gamma_a - 1)/gamma_a) - Tt3) / eta_hpc
    
    Pt5 = Pt4 * pi_b
    
    denom = (eta_b * LHV - cp_g * Tt5)
    if denom <= 0:
        return out
    f = (cp_g * Tt5 - cp_a * Tt4) / denom
    if f <= 0.001 or f >= 0.1:
        return out
    
    m_dot_fuel = f * m_dot_core
    m_dot_gas = m_dot_core * (1 + f)
    
    Tt6 = Tt5 - (cp_a * (Tt4 - Tt3)) / ((1 + f) * cp_g * eta_mhp)
    if Tt6 <= 0 or Tt6 >= Tt5:
        return out
    Pt6 = Pt5 * ((Tt5 - (Tt5 - Tt6) / eta_hpt) / Tt5)**(gamma_g / (gamma_g - 1))
    
    W_lp_comp = m_dot_fan * cp_a * (Tt10 - Tt2) + m_dot_core * cp_a * (Tt3 - Tt10)
    Tt7 = Tt6 - W_lp_comp / (m_dot_gas * cp_g * eta_mlp)
    if Tt7 <= 0 or Tt7 >= Tt6:
        return out
    Pt7 = Pt6 * ((Tt6 - (Tt6 - Tt7) / eta_lpt) / Tt6)**(gamma_g / (gamma_g - 1))
    
    Pt8 = Pt7
    Tt8 = Tt7
    
    NPR_fan  = Pt10 / Pa
    NPR_core = Pt8 / Pa
    if NPR_fan < 1.0 or NPR_core < 1.0:
        return out
    
    V11 = np.sqrt(max(0, 2 * cp_a * Tt10 * eta_fn * (1 - (1/NPR_fan)**((gamma_a - 1)/gamma_a))))
    V9  = np.sqrt(max(0, 2 * cp_g * Tt8 * eta_cn * (1 - (1/NPR_core)**((gamma_g - 1)/gamma_g))))
    
    F_core = m_dot_gas * V9
    F_byp  = m_dot_fan * V11
    F_ram  = m_dot_total * Va
    F_net  = F_core + F_byp - F_ram
    
    out['V9'] = V9
    out['V11'] = V11
    out['F_core'] = F_core
    out['F_byp'] = F_byp
    out['F_ram'] = F_ram
    out['F_net'] = F_net
    
    out['TSFC'] = (m_dot_fuel / F_net) * 1e6
    out['SFC'] = (m_dot_fuel * 3600) / (F_net / 1000)
    out['m_dot_fuel'] = m_dot_fuel
    out['f'] = f
    out['F_spec_total'] = F_net / m_dot_total
    out['Pa'] = Pa
    out['Ta'] = Ta
    
    P_fuel = m_dot_fuel * LHV
    P_jet  = 0.5 * m_dot_fan * (V11**2 - Va**2) + 0.5 * m_dot_gas * V9**2 - 0.5 * m_dot_core * Va**2
    W_prop = F_net * Va
    
    out['eta_th'] = (P_jet / P_fuel) * 100
    out['eta_p']  = (W_prop / P_jet) * 100
    out['eta_o']  = (W_prop / P_fuel) * 100
    
    M_st = np.array([0.45, 0.45, 0.40, 0.30, 0.25, 0.40, 0.45, 0.45])
    T_st = np.array([Tt2/(1+0.2*M_st[0]**2), Tt10/(1+0.2*M_st[1]**2), Tt3/(1+0.2*M_st[2]**2), Tt4/(1+0.2*M_st[3]**2), Tt5/(1+0.165*M_st[4]**2), Tt6/(1+0.165*M_st[5]**2), Tt7/(1+0.165*M_st[6]**2), Tt8/(1+0.165*M_st[7]**2)])
    P_st = np.array([Pt2/(1+0.2*M_st[0]**2)**3.5, Pt10/(1+0.2*M_st[1]**2)**3.5, Pt3/(1+0.2*M_st[2]**2)**3.5, Pt4/(1+0.2*M_st[3]**2)**3.5, Pt5/(1+0.165*M_st[4]**2)**4.03, Pt6/(1+0.165*M_st[5]**2)**4.03, Pt7/(1+0.165*M_st[6]**2)**4.03, Pt8/(1+0.165*M_st[7]**2)**4.03])
    V_st = np.array([M_st[0]*np.sqrt(gamma_a*R_a*T_st[0]), M_st[1]*np.sqrt(gamma_a*R_a*T_st[1]), M_st[2]*np.sqrt(gamma_a*R_a*T_st[2]), M_st[3]*np.sqrt(gamma_a*R_a*T_st[3]), M_st[4]*np.sqrt(gamma_g*R_g*T_st[4]), M_st[5]*np.sqrt(gamma_g*R_g*T_st[5]), M_st[6]*np.sqrt(gamma_g*R_g*T_st[6]), M_st[7]*np.sqrt(gamma_g*R_g*T_st[7])])
     
    out['st_names'] = ['a', '2', '10', '11', '3', '4', '5', '6', '7', '8', '9']
    out['st_desc']  = ['Freestream', 'Inlet/Fan In', 'Fan Out (Bypass)', 'Fan Nozzle Exit', 'LPC Out', 'HPC Out', 'HPT In', 'LPT In', 'LPT Out', 'Core Duct', 'Core Nozzle Exit']
    
    out['Pt_kPa']    = np.array([Pta, Pt2, Pt10, Pt10, Pt3, Pt4, Pt5, Pt6, Pt7, Pt8, Pt8]) / 1000
    out['P_kPa']     = np.array([Pa, P_st[0], P_st[1], Pa, P_st[2], P_st[3], P_st[4], P_st[5], P_st[6], P_st[7], Pa]) / 1000
    out['Tt_K']      = np.array([Tta, Tt2, Tt10, Tt10, Tt3, Tt4, Tt5, Tt6, Tt7, Tt8, Tt8])
    out['T_K']       = np.array([Ta, T_st[0], T_st[1], Tt10-(V11**2)/(2*cp_a), T_st[2], T_st[3], T_st[4], T_st[5], T_st[6], T_st[7], Tt8-(V9**2)/(2*cp_g)])
    out['Vel_ms']    = np.array([Va, V_st[0], V_st[1], V11, V_st[2], V_st[3], V_st[4], V_st[5], V_st[6], V_st[7], V9])
    
    return out

# --- Helper Functions for Maps and Sweeps ---
def get_map_vector(type_str):
    if type_str in ['TIT', 'Turbine Inlet Temp (TIT)']: return np.linspace(1150, 1650, 18), 'TIT (K)'
    elif type_str in ['\\pi_f', 'Fan Pressure Ratio (\\pi_f)']: return np.linspace(1.2, 2.1, 18), '\\pi_f'
    elif type_str in ['\\pi_{lpc}', 'LPC Pressure Ratio (\\pi_{lpc})']: return np.linspace(1.1, 2.1, 18), '\\pi_{lpc}'
    elif type_str in ['BPR', 'Bypass Ratio (BPR)']: return np.linspace(1.5, 8.0, 18), 'BPR'
    elif type_str in ['\\pi_{hpc}', 'HPC Pressure Ratio', 'HPC Pressure Ratio (\\pi_{hpc})']: return np.linspace(6.0, 18.0, 18), '\\pi_{hpc}'
    elif type_str in ['Alt', 'Altitude (Alt)']: return np.linspace(0, 35000, 18), 'Altitude (ft)'
    elif type_str in ['\\eta_f', 'Fan Efficiency (\\eta_f)']: return np.linspace(0.88, 0.98, 18), '\\eta_f'
    elif type_str in ['\\eta_{lpc}', 'LPC Efficiency (\\eta_{lpc})']: return np.linspace(0.88, 0.98, 18), '\\eta_{lpc}'
    elif type_str in ['\\eta_{hpc}', 'HPC Efficiency (\\eta_{hpc})']: return np.linspace(0.88, 0.98, 18), '\\eta_{hpc}'
    elif type_str in ['\\eta_{hpt}', 'HPT Efficiency (\\eta_{hpt})']: return np.linspace(0.88, 0.98, 18), '\\eta_{hpt}'
    elif type_str in ['\\eta_{lpt}', 'LPT Efficiency (\\eta_{lpt})']: return np.linspace(0.88, 0.98, 18), '\\eta_{lpt}'
    return np.linspace(1.0, 2.0, 18), type_str

def plot_styled_contour(ax, X, Y, Z, lblX, lblY, titleStr):
    ax.clear()
    if np.isnan(Z).all():
        ax.set_title(f"{titleStr} (Out of Bounds)", fontsize=9, fontweight='bold', color='#f8fafc')
        return
    
    if "Thrust" in titleStr: cmap_choice = 'CMRmap'
    elif "TSFC" in titleStr: cmap_choice = 'copper'
    elif "Thermal" in titleStr: cmap_choice = 'ocean'
    else: cmap_choice = 'gist_earth'

    cs_fill = ax.contourf(X, Y, Z, levels=15, cmap=cmap_choice, alpha=0.92, antialiased=True)
    cbar = plt.colorbar(cs_fill, ax=ax)
    cbar.ax.tick_params(labelsize=6, colors='#cbd5e1')
    
    cs_lines = ax.contour(X, Y, Z, levels=10, colors='#ffffff', linewidths=0.8, alpha=0.75)
    ax.clabel(cs_lines, inline=True, fmt='%.2f', fontsize=7, colors='white')
    
    ax.set_title(titleStr, fontsize=9, fontweight='bold', color='#38bdf8')
    ax.set_xlabel(lblX, fontsize=8, color='#94a3b8')
    ax.set_ylabel(lblY, fontsize=8, color='#94a3b8')
    ax.tick_params(axis='both', labelsize=7, colors='#94a3b8')
    ax.grid(True, linestyle='--', alpha=0.25, color='#475569')

# --- Sidebar Inputs Control Panel ---
st.sidebar.title("⚙️ Engine Control Panel")
st.sidebar.markdown("---")

with st.sidebar.expander("🌍 Ambient & Flight", expanded=True):
    alt_val = st.number_input("Altitude (ft)", value=0.0, step=500.0, format="%.1f")
    ma_val = st.number_input("Mach Number ($M_a$)", value=0.2, step=0.05, format="%.2f")
    mcore_val = st.number_input("Core Mass Flow (kg/s)", value=6.0, step=0.5, format="%.1f")

with st.sidebar.expander("🔧 Hardware Specs", expanded=True):
    tit_val = st.number_input("TIT (K)", value=1350.0, step=10.0, format="%.1f")
    bpr_val = st.number_input("Bypass Ratio (BPR)", value=4.17, step=0.1, format="%.2f")
    pif_val = st.number_input("Fan Pressure Ratio ($\pi_f$)", value=1.78, step=0.01, format="%.2f")
    pilpc_val = st.number_input("LPC Pressure Ratio ($\pi_{lpc}$)", value=1.35, step=0.01, format="%.2f")
    pihpc_val = st.number_input("HPC Pressure Ratio ($\pi_{hpc}$)", value=11.0, step=0.5, format="%.2f")

with st.sidebar.expander("💨 Fan Aero Sizing", expanded=False):
    rpm_val = st.number_input("Rotational Speed (RPM)", value=9500.0, step=100.0, format="%.1f")
    hub_tip_val = st.number_input("Hub-Tip Ratio ($\nu$)", value=0.35, step=0.01, format="%.2f")
    mz2_val = st.number_input("Axial Mach Number", value=0.32, step=0.01, format="%.2f")
    ar_val = st.number_input("Blade Aspect Ratio", value=2.5, step=0.1, format="%.2f")

# Constants Initialization
LHV = 45.0e6
eta_d = 0.98; eta_f_base = 0.93; eta_lpc_base = 0.93; eta_hpc_base = 0.93
eta_b = 0.99; pi_b = 0.97; eta_hpt_base = 0.93; eta_lpt_base = 0.93
eta_mhp = 0.99; eta_mlp = 0.99; eta_fn = 0.98; eta_cn = 0.98
gamma_a = 1.40; R_a = 287; cp_a = (gamma_a * R_a) / (gamma_a - 1)
gamma_g = 1.33; R_g = 287; cp_g = (gamma_g * R_g) / (gamma_g - 1)

# Baseline Run
base = run_cycle(alt_val, tit_val, pif_val, pilpc_val, pihpc_val, bpr_val, ma_val, mcore_val, LHV, eta_d, eta_f_base, eta_lpc_base, eta_hpc_base, eta_b, pi_b, eta_hpt_base, eta_lpt_base, eta_mhp, eta_mlp, eta_fn, eta_cn, gamma_a, R_a, cp_a, gamma_g, R_g, cp_g)

if np.isnan(base['F_net']):
    st.error("The combination of selected baseline parameters resulted in an invalid thermodynamic cycle.")
else:
    m_dot_fan_total = base['m_dot_core'] * (1 + bpr_val)
    Tt2 = base['Tt_K'][2]; Pt2 = base['Pt_kPa'][2] * 1000
    
    T2 = Tt2 / (1 + (gamma_a - 1)/2 * mz2_val**2)
    P2 = Pt2 / (1 + (gamma_a - 1)/2 * mz2_val**2)**(gamma_a / (gamma_a - 1))
    rho2 = P2 / (R_a * T2)
    V_z2 = mz2_val * np.sqrt(gamma_a * R_a * T2)
    a2   = np.sqrt(gamma_a * R_a * T2)
    
    A_annulus = m_dot_fan_total / (rho2 * V_z2)
    D_tip = np.sqrt((4 * A_annulus) / (np.pi * (1 - hub_tip_val**2)))
    D_hub = hub_tip_val * D_tip
    D_mean = (D_tip + D_hub) / 2
    h_blade = (D_tip - D_hub) / 2
    
    omega = (2 * np.pi * rpm_val) / 60
    U_tip  = omega * (D_tip / 2)
    U_mean = omega * (D_mean / 2)
    
    M_rel_tip  = np.sqrt(V_z2**2 + U_tip**2) / a2
    M_rel_mean = np.sqrt(V_z2**2 + U_mean**2) / a2
    c_blade = h_blade / ar_val

    # --- Tabs Layout ---
    tabSummary, tabProfiles, tabTS, tab1D, tab2D = st.tabs([
        "Station Summary", "Station Profiles", "T-s Diagram", "1D Sweeps", "2D Maps"
    ])
    
    # --- Tab 1: Station Summary ---
    with tabSummary:
        st.subheader("Engine Stations Diagram")
        
        _, col_img, _ = st.columns([1, 2, 1])
        with col_img:
            try:
                st.image("turbofan_stations.png", caption="Turbofan Engine Station Numbering Schema", use_container_width=True)
            except Exception:
                st.info("💡 لظهور الرسم التوضيحي، تأكد من وضع صورة المحطة في نفس مجلد ملف الـ script باسم 'turbofan_stations.png'")
            
        st.subheader("Station Summary & Parameters Table")
        tData = []
        for i in range(len(base['st_names'])):
            tData.append({
                "Station": base['st_names'][i],
                "Engine Section": base['st_desc'][i],
                "Pt (kPa)": f"{base['Pt_kPa'][i]:.2f}",
                "P_static (kPa)": f"{base['P_kPa'][i]:.2f}",
                "Tt (K)": f"{base['Tt_K'][i]:.2f}",
                "T_static (K)": f"{base['T_K'][i]:.2f}",
                "Velocity (m/s)": f"{base['Vel_ms'][i]:.2f}"
            })
        st.table(tData)
        
        # CSV Download Button Export Feature
        csv_data = "Station,Engine Section,Pt_kPa,P_static_kPa,Tt_K,T_static_K,Velocity_ms\n"
        for i in range(len(base['st_names'])):
            csv_data += f"{base['st_names'][i]},{base['st_desc'][i]},{base['Pt_kPa'][i]:.2f},{base['P_kPa'][i]:.2f},{base['Tt_K'][i]:.2f},{base['T_K'][i]:.2f},{base['Vel_ms'][i]:.2f}\n"
        st.download_button(
            label="📥 Download Station Table as CSV",
            data=csv_data,
            file_name="turbofan_station_summary.csv",
            mime="text/csv"
        )
        
        st.subheader("Detailed Performance & Sizing Output")
        perfText = (
            f"Ambient Conditions         : Altitude: {alt_val:.0f} ft | Pa: {base['Pa']/1000:.2f} kPa | Ta: {base['Ta']:.2f} K\n"
            f"Actual Operating Mass Flow  : Core: {base['m_dot_core']:.3f} kg/s | Total: {base['m_dot_core']*(1+bpr_val):.3f} kg/s\n"
            f"--- Fan Geometry & Aerodynamics ---\n"
            f"Fan Tip Diameter (D_tip)    : {D_tip:.4f} m ({D_tip*39.3701:.2f} in)\n"
            f"Fan Hub Diameter (D_hub)    : {D_hub:.4f} m ({D_hub*39.3701:.2f} in)\n"
            f"Fan Mean Diameter (D_mean)  : {D_mean:.4f} m ({D_mean*39.3701:.2f} in)\n"
            f"Fan Blade Height (h)        : {h_blade:.4f} m ({h_blade*39.3701:.2f} in)\n"
            f"Blade Aspect Ratio (AR)     : {ar_val:.2f} | Blade Chord: {c_blade:.4f} m\n"
            f"--- Kinematics & Aerodynamic Speed (RPM: {rpm_val:.0f}) ---\n"
            f"Blade Tip Speed (U_tip)     : {U_tip:.2f} m/s\n"
            f"Tip Relative Mach (M_rel,tip): {M_rel_tip:.3f}\n"
            f"Mean Relative Mach (M_rel,m) : {M_rel_mean:.3f}\n"
            f"--- Thrust Components ---\n"
            f"Core Gross Thrust           : {base['F_core']:.2f} N ({base['F_core']/1000:.2f} kN)\n"
            f"Bypass Gross Thrust         : {base['F_byp']:.2f} N ({base['F_byp']/1000:.2f} kN)\n"
            f"Ram Drag                    : {base['F_ram']:.2f} N ({base['F_ram']/1000:.2f} kN)\n"
            f"Net Thrust (F_net)          : {base['F_net']:.2f} N ({base['F_net']/1000:.2f} kN)\n"
            f"--- Engine Performance ---\n"
            f"Specific Thrust (Total Air) : {base['F_spec_total']:.4f} N/(kg/s)\n"
            f"TSFC                        : {base['TSFC']:.4f} mg/N/s | SFC: {base['SFC']:.4f} kg/(kN·h)\n"
            f"Fuel Flow Rate (m_dot_fuel) : {base['m_dot_fuel']:.4f} kg/s   | f: {base['f']:.4f}\n"
            f"Efficiencies                : Thermal: {base['eta_th']:.2f}% | Propulsive: {base['eta_p']:.2f}% | Overall: {base['eta_o']:.2f}%"
        )
        st.code(perfText, language="text")

    # --- Tab 2: Station Profiles ---
    with tabProfiles:
        st.subheader("Station Profiles (Temperature & Pressure)")
        c_idx = [0, 1, 4, 5, 6, 7, 8, 9, 10]
        labels_core = [base['st_names'][i] for i in c_idx]
        b_idx = [0, 1, 2, 3]
        
        fig_prof, (axTemp, axPress) = plt.subplots(2, 1, figsize=(8, 5))
        
        axTemp.plot(range(len(c_idx)), base['Tt_K'][c_idx], color='#f43f5e', marker='o', linewidth=1.8, label='Core T_t')
        axTemp.plot(range(len(c_idx)), base['T_K'][c_idx], color='#fb923c', marker='s', linestyle='--', linewidth=1.4, label='Core T')
        axTemp.plot(range(len(b_idx)), base['Tt_K'][b_idx], color='#c084fc', marker='^', linestyle='-', linewidth=1.4, label='Bypass T_t,byp')
        axTemp.set_xticks(range(len(c_idx)))
        axTemp.set_xticklabels(labels_core, fontsize=8, color='#cbd5e1')
        axTemp.tick_params(axis='y', labelsize=8, colors='#cbd5e1')
        axTemp.grid(True, linestyle='--', alpha=0.3, color='#475569')
        axTemp.set_title('Temperature Variation Across Engine Stations', fontsize=10, fontweight='bold', color='#38bdf8')
        axTemp.set_ylabel('Temperature (K)', fontsize=8, color='#38bdf8')
        axTemp.legend(fontsize=7, facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')
        
        axPress.plot(range(len(c_idx)), base['Pt_kPa'][c_idx], color='#38bdf8', marker='o', linewidth=1.8, label='Core P_t')
        axPress.plot(range(len(c_idx)), base['P_kPa'][c_idx], color='#38bdf8', marker='s', linestyle='--', linewidth=1.4, label='Core P')
        axPress.plot(range(len(b_idx)), base['Pt_kPa'][b_idx], color='#2dd4bf', marker='^', linestyle='-', linewidth=1.4, label='Bypass P_t,byp')
        axPress.set_xticks(range(len(c_idx)))
        axPress.set_xticklabels(labels_core, fontsize=8, color='#cbd5e1')
        axPress.tick_params(axis='y', labelsize=8, colors='#cbd5e1')
        axPress.grid(True, linestyle='--', alpha=0.3, color='#475569')
        axPress.set_title('Pressure Variation Across Engine Stations', fontsize=10, fontweight='bold', color='#38bdf8')
        axPress.set_ylabel('Pressure (kPa)', fontsize=8, color='#38bdf8')
        axPress.legend(fontsize=7, facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')
        
        plt.tight_layout()
        st.pyplot(fig_prof)

    # --- Tab 3: T-s Diagram ---
    with tabTS:
        st.subheader("Temperature–Entropy (T–s) Diagram")
        c_ts = [0, 1, 4, 5, 6, 7, 8, 9, 10]
        T_core = base['Tt_K'][c_ts]; P_core = base['Pt_kPa'][c_ts]
        s_core = np.zeros(len(c_ts))
        for i in range(1, len(c_ts)):
            if i <= 3:
                ds = cp_a * np.log(T_core[i]/T_core[i-1]) - R_a * np.log(P_core[i]/P_core[i-1])
            else:
                ds = cp_g * np.log(T_core[i]/T_core[i-1]) - R_g * np.log(P_core[i]/P_core[i-1])
            s_core[i] = s_core[i-1] + ds
            
        T_byp = base['Tt_K'][b_idx]; P_byp = base['Pt_kPa'][b_idx]
        s_byp = np.zeros(len(b_idx))
        for i in range(1, len(b_idx)):
            s_byp[i] = s_byp[i-1] + cp_a * np.log(T_byp[i]/T_byp[i-1]) - R_a * np.log(P_byp[i]/P_byp[i-1])
            
        fig_ts, axTS = plt.subplots(figsize=(7, 4))
        axTS.plot(s_core, T_core, color='#fb7185', marker='o', linewidth=1.8, label='Core Flow Path')
        axTS.plot(s_byp, T_byp, color='#60a5fa', marker='s', linestyle='--', linewidth=1.6, label='Bypass Flow Path')
        axTS.grid(True, linestyle='--', alpha=0.3, color='#475569')
        axTS.set_title('Temperature–Entropy (T–s) Diagram', fontsize=10, fontweight='bold', color='#38bdf8')
        axTS.set_xlabel('Delta s (J/kg·K)', fontsize=8, fontweight='bold', color='#38bdf8')
        axTS.set_ylabel('Total Temperature T_t (K)', fontsize=8, fontweight='bold', color='#38bdf8')
        axTS.tick_params(axis='both', labelsize=8, colors='#cbd5e1')
        axTS.legend(fontsize=8, facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')
        st.pyplot(fig_ts)

    # --- Tab 4: 1D Sweeps ---
    with tab1D:
        st.subheader("1D Parametric Sweeps Analysis")
        sweep_var = st.selectbox("Select Parameter to Sweep:", [
            'Fan Pressure Ratio (pi_f)', 'LPC Pressure Ratio (pi_lpc)', 'HPC Pressure Ratio (pi_hpc)',
            'Bypass Ratio (BPR)', 'Turbine Inlet Temp (TIT)', 'Altitude (Alt)',
            'Fan Efficiency (eta_f)', 'LPC Efficiency (eta_lpc)', 'HPC Efficiency (eta_hpc)',
            'HPT Efficiency (eta_hpt)', 'LPT Efficiency (eta_lpt)'
        ])
        
        mapping_1d = {
            'Fan Pressure Ratio (pi_f)': ('pi_f', np.linspace(1.1, 2.1, 40), 'pi_f'),
            'LPC Pressure Ratio (pi_lpc)': ('pi_lpc', np.linspace(1.1, 2.1, 40), 'pi_lpc'),
            'HPC Pressure Ratio (pi_hpc)': ('pi_hpc', np.linspace(8.0, 16.0, 40), 'pi_hpc'),
            'Bypass Ratio (BPR)': ('bpr', np.linspace(1.0, 6.0, 40), 'BPR'),
            'Turbine Inlet Temp (TIT)': ('tit', np.linspace(1200, 1600, 40), 'TIT (K)'),
            'Altitude (Alt)': ('alt', np.linspace(0, 35000, 40), 'Altitude (ft)'),
            'Fan Efficiency (eta_f)': ('eta_f', np.linspace(0.88, 0.98, 40), 'eta_f'),
            'LPC Efficiency (eta_lpc)': ('eta_lpc', np.linspace(0.88, 0.98, 40), 'eta_lpc'),
            'HPC Efficiency (eta_hpc)': ('eta_hpc', np.linspace(0.88, 0.98, 40), 'eta_hpc'),
            'HPT Efficiency (eta_hpt)': ('eta_hpt', np.linspace(0.88, 0.98, 40), 'eta_hpt'),
            'LPT Efficiency (eta_lpt)': ('eta_lpt', np.linspace(0.88, 0.98, 40), 'eta_lpt')
        }
        
        s_type, s_range, x_lbl = mapping_1d[sweep_var]
        
        res_F, res_TSFC, res_th, res_p, res_o = [], [], [], [], []
        for val in s_range:
            alt, t, pf, pl, ph, b, ef, elpc, ehpc, ehpt, elpt = alt_val, tit_val, pif_val, pilpc_val, pihpc_val, bpr_val, eta_f_base, eta_lpc_base, eta_hpc_base, eta_hpt_base, eta_lpt_base
            if s_type == 'pi_f': pf = val
            elif s_type == 'pi_lpc': pl = val
            elif s_type == 'pi_hpc': ph = val
            elif s_type == 'bpr': b = val
            elif s_type == 'tit': t = val
            elif s_type == 'alt': alt = val
            elif s_type == 'eta_f': ef = val
            elif s_type == 'eta_lpc': elpc = val
            elif s_type == 'eta_hpc': ehpc = val
            elif s_type == 'eta_hpt': ehpt = val
            elif s_type == 'eta_lpt': elpt = val
            
            out = run_cycle(alt, t, pf, pl, ph, b, ma_val, mcore_val, LHV, eta_d, ef, elpc, ehpc, eta_b, pi_b, ehpt, elpt, eta_mhp, eta_mlp, eta_fn, eta_cn, gamma_a, R_a, cp_a, gamma_g, R_g, cp_g)
            if np.isnan(out['F_net']) or out['F_net'] <= 0:
                res_F.append(np.nan); res_TSFC.append(np.nan); res_th.append(np.nan); res_p.append(np.nan); res_o.append(np.nan)
            else:
                res_F.append(out['F_net']/1000); res_TSFC.append(out['TSFC']); res_th.append(out['eta_th']); res_p.append(out['eta_p']); res_o.append(out['eta_o'])
                
        fig_1d, axs = plt.subplots(2, 3, figsize=(11, 5))
        axs[0, 0].plot(s_range, res_F, color='#38bdf8', linewidth=1.8); axs[0, 0].grid(True, alpha=0.3); axs[0, 0].set_title('Thrust (kN)', fontsize=9, fontweight='bold', color='#38bdf8'); axs[0, 0].set_xlabel(x_lbl, fontsize=8, color='#94a3b8'); axs[0, 0].tick_params(labelsize=7, colors='#94a3b8')
        axs[0, 1].plot(s_range, res_TSFC, color='#f87171', linewidth=1.8); axs[0, 1].grid(True, alpha=0.3); axs[0, 1].set_title('TSFC (mg/N/s)', fontsize=9, fontweight='bold', color='#38bdf8'); axs[0, 1].set_xlabel(x_lbl, fontsize=8, color='#94a3b8'); axs[0, 1].tick_params(labelsize=7, colors='#94a3b8')
        axs[0, 2].plot(s_range, res_th, color='#4ade80', linewidth=1.8); axs[0, 2].grid(True, alpha=0.3); axs[0, 2].set_title('Thermal Eff. (%)', fontsize=9, fontweight='bold', color='#38bdf8'); axs[0, 2].set_xlabel(x_lbl, fontsize=8, color='#94a3b8'); axs[0, 2].tick_params(labelsize=7, colors='#94a3b8')
        axs[1, 0].plot(s_range, res_p, color='#c084fc', linewidth=1.8); axs[1, 0].grid(True, alpha=0.3); axs[1, 0].set_title('Propulsive Eff. (%)', fontsize=9, fontweight='bold', color='#38bdf8'); axs[1, 0].set_xlabel(x_lbl, fontsize=8, color='#94a3b8'); axs[1, 0].tick_params(labelsize=7, colors='#94a3b8')
        axs[1, 1].plot(s_range, res_o, color='#2dd4bf', linewidth=1.8); axs[1, 1].grid(True, alpha=0.3); axs[1, 1].set_title('Overall Eff. (%)', fontsize=9, fontweight='bold', color='#38bdf8'); axs[1, 1].set_xlabel(x_lbl, fontsize=8, color='#94a3b8'); axs[1, 1].tick_params(labelsize=7, colors='#94a3b8')
        axs[1, 2].axis('off')
        plt.tight_layout()
        st.pyplot(fig_1d)

    # --- Tab 5: 2D Contour Maps ---
    with tab2D:
        st.subheader("2D Performance Contour Maps")
        colX, colY = st.columns(2)
        items2D = ['TIT', '\\pi_f', '\\pi_{lpc}', 'BPR', '\\pi_{hpc}', 'Alt', '\\eta_f', '\\eta_{lpc}', '\\eta_{hpc}', '\\eta_{hpt}', '\\eta_{lpt}']
        
        with colX:
            varX = st.selectbox("X-Axis Parameter:", items2D, index=0)
        with colY:
            varY = st.selectbox("Y-Axis Parameter:", items2D, index=1)
            
        vecX, lblX = get_map_vector(varX)
        vecY, lblY = get_map_vector(varY)
        
        X_mesh, Y_mesh = np.meshgrid(vecX, vecY)
        Z_F = np.zeros_like(X_mesh)
        Z_TSFC = np.zeros_like(X_mesh)
        Z_th = np.zeros_like(X_mesh)
        Z_o = np.zeros_like(X_mesh)
        
        for i in range(len(vecX)):
            for j in range(len(vecY)):
                alt, t, pf, pl, ph, b, ef, elpc, ehpc, ehpt, elpt = alt_val, tit_val, pif_val, pilpc_val, pihpc_val, bpr_val, eta_f_base, eta_lpc_base, eta_hpc_base, eta_hpt_base, eta_lpt_base
                
                vx = vecX[i]
                if varX in ['TIT', 'Turbine Inlet Temp (TIT)']: t = vx
                elif varX in ['\\pi_f', 'Fan Pressure Ratio (\\pi_f)']: pf = vx
                elif varX in ['\\pi_{lpc}', 'LPC Pressure Ratio (\\pi_{lpc})']: pl = vx
                elif varX in ['BPR', 'Bypass Ratio (BPR)']: b = vx
                elif varX in ['\\pi_{hpc}', 'HPC Pressure Ratio']: ph = vx
                elif varX in ['Alt', 'Altitude (Alt)']: alt = vx
                elif varX in ['\\eta_f', 'Fan Efficiency (\\eta_f)']: ef = vx
                elif varX in ['\\eta_{lpc}', 'LPC Efficiency (\\eta_{lpc})']: elpc = vx
                elif varX in ['\\eta_{hpc}', 'HPC Efficiency (\\eta_{hpc})']: ehpc = vx
                elif varX in ['\\eta_{hpt}', 'HPT Efficiency (\\eta_{hpt})']: ehpt = vx
                elif varX in ['\\eta_{lpt}', 'LPT Efficiency (\\eta_{lpt})']: elpt = vx
                
                vy = vecY[j]
                if varY in ['TIT', 'Turbine Inlet Temp (TIT)']: t = vy
                elif varY in ['\\pi_f', 'Fan Pressure Ratio (\\pi_f)']: pf = vy
                elif varY in ['\\pi_{lpc}', 'LPC Pressure Ratio (\\pi_{lpc})']: pl = vy
                elif varY in ['BPR', 'Bypass Ratio (BPR)']: b = vy
                elif varY in ['\\pi_{hpc}', 'HPC Pressure Ratio']: ph = vy
                elif varY in ['Alt', 'Altitude (Alt)']: alt = vy
                elif varY in ['\\eta_f', 'Fan Efficiency (\\eta_f)']: ef = vy
                elif varY in ['\\eta_{lpc}', 'LPC Efficiency (\\eta_{lpc})']: elpc = vy
                elif varY in ['\\eta_{hpc}', 'HPC Efficiency (\\eta_{hpc})']: ehpc = vy
                elif varY in ['\\eta_{hpt}', 'HPT Efficiency (\\eta_{hpt})']: ehpt = vy
                elif varY in ['\\eta_{lpt}', 'LPT Efficiency (\\eta_{lpt})']: elpt = vy
                
                out = run_cycle(alt, t, pf, pl, ph, b, ma_val, mcore_val, LHV, eta_d, ef, elpc, ehpc, eta_b, pi_b, ehpt, elpt, eta_mhp, eta_mlp, eta_fn, eta_cn, gamma_a, R_a, cp_a, gamma_g, R_g, cp_g)
                if np.isnan(out['F_net']) or out['F_net'] <= 0:
                    Z_F[j, i] = np.nan; Z_TSFC[j, i] = np.nan; Z_th[j, i] = np.nan; Z_o[j, i] = np.nan
                else:
                    Z_F[j, i] = out['F_net'] / 1000
                    Z_TSFC[j, i] = out['TSFC']
                    Z_th[j, i] = out['eta_th']
                    Z_o[j, i] = out['eta_o']
                    
        fig_2d, axs_2d = plt.subplots(2, 2, figsize=(9, 7))
        plot_styled_contour(axs_2d[0, 0], X_mesh, Y_mesh, Z_F, lblX, lblY, 'Net Thrust (kN)')
        plot_styled_contour(axs_2d[0, 1], X_mesh, Y_mesh, Z_TSFC, lblX, lblY, 'TSFC (mg/N/s)')
        plot_styled_contour(axs_2d[1, 0], X_mesh, Y_mesh, Z_th, lblX, lblY, 'Thermal Efficiency (%)')
        plot_styled_contour(axs_2d[1, 1], X_mesh, Y_mesh, Z_o, lblX, lblY, 'Overall Efficiency (%)')
        
        plt.tight_layout()
        st.pyplot(fig_2d)