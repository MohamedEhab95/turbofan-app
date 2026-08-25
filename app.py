import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration & Light MATLAB-Style Styling ---
st.set_page_config(
    page_title="Turbofan Cycle Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Switch Matplotlib defaults to Light Mode matching MATLAB figures
plt.style.use('default')
plt.rcParams.update({
    "figure.facecolor": "#f0f2f5",
    "axes.facecolor": "#ffffff",
    "savefig.facecolor": "#f0f2f5",
    "text.color": "#111827",
    "axes.labelcolor": "#111827",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "axes.edgecolor": "#9ca3af"
})

st.markdown("""
    <style>
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

# --- Fan Geometry Calculation Function ---
def calculate_fan_geometry_and_kinematics(base_out, bpr_val, rpm_fan, mz2_val, nu_hub_tip, ar_blade, gamma_a=1.40, R_a=287):
    m_dot_fan_total = base_out['m_dot_core'] * (1 + bpr_val)
    Tt2 = base_out['Tt_K'][1]
    Pt2 = base_out['Pt_kPa'][1] * 1000
    
    T2 = Tt2 / (1 + (gamma_a - 1) / 2 * mz2_val**2)
    P2 = Pt2 / (1 + (gamma_a - 1) / 2 * mz2_val**2)**(gamma_a / (gamma_a - 1))
    rho2 = P2 / (R_a * T2)
    V_z2 = mz2_val * np.sqrt(gamma_a * R_a * T2)
    a2 = np.sqrt(gamma_a * R_a * T2)
    
    A_annulus = m_dot_fan_total / (rho2 * V_z2)
    D_tip = np.sqrt((4 * A_annulus) / (np.pi * (1 - nu_hub_tip**2)))
    D_hub = nu_hub_tip * D_tip
    D_mean = (D_tip + D_hub) / 2
    h_blade = (D_tip - D_hub) / 2
    
    omega = (2 * np.pi * rpm_fan) / 60
    U_tip = omega * (D_tip / 2)
    U_mean = omega * (D_mean / 2)
    
    M_rel_tip = np.sqrt(V_z2**2 + U_tip**2) / a2
    M_rel_mean = np.sqrt(V_z2**2 + U_mean**2) / a2
    c_blade = h_blade / ar_blade
    
    return {
        'D_tip': D_tip, 'D_hub': D_hub, 'D_mean': D_mean, 'h_blade': h_blade,
        'U_tip': U_tip, 'M_rel_tip': M_rel_tip, 'M_rel_mean': M_rel_mean,
        'c_blade': c_blade, 'A_annulus': A_annulus
    }

def get_map_vector(type_str):
    if type_str in ['TIT', 'Turbine Inlet Temp (TIT)']: return np.linspace(1150, 1650, 18), 'TIT (K)'
    elif type_str in ['\\pi_f', 'Fan Pressure Ratio (\\pi_f)']: return np.linspace(1.2, 2.1, 18), '$\\pi_f$'
    elif type_str in ['\\pi_{lpc}', 'LPC Pressure Ratio (\\pi_{lpc})']: return np.linspace(1.1, 2.1, 18), '$\\pi_{lpc}$'
    elif type_str in ['BPR', 'Bypass Ratio (BPR)']: return np.linspace(1.5, 8.0, 18), 'BPR'
    elif type_str in ['\\pi_{hpc}', 'HPC Pressure Ratio', 'HPC Pressure Ratio (\\pi_{hpc})']: return np.linspace(6.0, 18.0, 18), '$\\pi_{hpc}$'
    elif type_str in ['Alt', 'Altitude (Alt)']: return np.linspace(0, 35000, 18), 'Altitude (ft)'
    elif type_str in ['\\eta_f', 'Fan Efficiency (\\eta_f)']: return np.linspace(0.88, 0.98, 18), '$\\eta_f$'
    elif type_str in ['\\eta_{lpc}', 'LPC Efficiency (\\eta_{lpc})']: return np.linspace(0.88, 0.98, 18), '$\\eta_{lpc}$'
    elif type_str in ['\\eta_{hpc}', 'HPC Efficiency (\\eta_{hpc})']: return np.linspace(0.88, 0.98, 18), '$\\eta_{hpc}$'
    elif type_str in ['\\eta_{hpt}', 'HPT Efficiency (\\eta_{hpt})']: return np.linspace(0.88, 0.98, 18), '$\\eta_{hpt}$'
    elif type_str in ['\\eta_{lpt}', 'LPT Efficiency (\\eta_{lpt})']: return np.linspace(0.88, 0.98, 18), '$\\eta_{lpt}$'
    return np.linspace(1.0, 2.0, 18), type_str

def plot_styled_contour(ax, X, Y, Z, lblX, lblY, titleStr):
    ax.clear()
    if np.isnan(Z).all():
        ax.set_title(f"{titleStr} (Out of Bounds)", fontsize=10, fontweight='bold', color='#111827')
        return
    
    cs_fill = ax.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.95, antialiased=True)
    cbar = plt.colorbar(cs_fill, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors='#111827')
    
    cs_lines = ax.contour(X, Y, Z, levels=20, colors='#000000', linewidths=0.9, alpha=0.85)
    ax.clabel(cs_lines, inline=True, fmt='%.4g', fontsize=7.5, colors='black')
    
    ax.set_title(titleStr, fontsize=11, fontweight='bold', color='#111827')
    ax.set_xlabel(lblX, fontsize=10, fontweight='bold', color='#111827')
    ax.set_ylabel(lblY, fontsize=10, fontweight='bold', color='#111827')
    ax.tick_params(axis='both', labelsize=9, colors='#111827')
    ax.grid(True, linestyle='--', alpha=0.3, color='#9ca3af')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

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

LHV = 45.0e6
eta_d = 0.98; eta_f_base = 0.93; eta_lpc_base = 0.93; eta_hpc_base = 0.93
eta_b = 0.99; pi_b = 0.97; eta_hpt_base = 0.93; eta_lpt_base = 0.93
eta_mhp = 0.99; eta_mlp = 0.99; eta_fn = 0.98; eta_cn = 0.98
gamma_a = 1.40; R_a = 287; cp_a = (gamma_a * R_a) / (gamma_a - 1)
gamma_g = 1.33; R_g = 287; cp_g = (gamma_g * R_g) / (gamma_g - 1)

base = run_cycle(alt_val, tit_val, pif_val, pilpc_val, pihpc_val, bpr_val, ma_val, mcore_val, LHV, eta_d, eta_f_base, eta_lpc_base, eta_hpc_base, eta_b, pi_b, eta_hpt_base, eta_lpt_base, eta_mhp, eta_mlp, eta_fn, eta_cn, gamma_a, R_a, cp_a, gamma_g, R_g, cp_g)

if np.isnan(base['F_net']):
    st.error("The combination of selected baseline parameters resulted in an invalid thermodynamic cycle.")
else:
    fan_res = calculate_fan_geometry_and_kinematics(base, bpr_val, rpm_val, mz2_val, hub_tip_val, ar_val, gamma_a, R_a)
    D_tip = fan_res['D_tip']; D_hub = fan_res['D_hub']; D_mean = fan_res['D_mean']; h_blade = fan_res['h_blade']
    U_tip = fan_res['U_tip']; M_rel_tip = fan_res['M_rel_tip']; M_rel_mean = fan_res['M_rel_mean']; c_blade = fan_res['c_blade']

    # --- Tabs Layout ---
    tabSummary, tabProfiles, tabTS, tab1D, tab2D = st.tabs([
        "Station Summary", "Station Profiles", "T-s Diagram", "1D Sweeps", "2D Maps"
    ])
    
    # --- Tab 1: Station Summary ---
    with tabSummary:
        # Side-by-side layout for Schematic and Summary Table
        col_schematic, col_table = st.columns([1, 1])
        
        with col_schematic:
            st.subheader("Standard Turbofan Engine Station Numbering Schematic")
            try:
                st.image("turbofan_stations.png", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load image. Make sure 'turbofan_stations.png' is in the same folder. Error: {e}")
                
        with col_table:
            st.subheader("Station Summary & Parameters Table")
            tData = []
            for i in range(len(base['st_names'])):
                tData.append({
                    "Station": base['st_names'][i],
                    "Section": base['st_desc'][i],
                    "Pt (kPa)": f"{base['Pt_kPa'][i]:.2f}",
                    "P (kPa)": f"{base['P_kPa'][i]:.2f}",
                    "Tt (K)": f"{base['Tt_K'][i]:.2f}",
                    "T (K)": f"{base['T_K'][i]:.2f}",
                    "V (m/s)": f"{base['Vel_ms'][i]:.2f}"
                })
            st.table(tData)
        
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
        
        # Categorized Tables for Performance & Sizing Output
        perf_categories = {
            "🌍 Ambient Conditions": [
                {"Parameter": "Altitude", "Value": f"{alt_val:.0f}", "Unit": "ft"},
                {"Parameter": "Ambient Pressure (Pa)", "Value": f"{base['Pa']/1000:.2f}", "Unit": "kPa"},
                {"Parameter": "Ambient Temperature (Ta)", "Value": f"{base['Ta']:.2f}", "Unit": "K"}
            ],
            "💨 Mass Flow Rates": [
                {"Parameter": "Core Mass Flow", "Value": f"{base['m_dot_core']:.3f}", "Unit": "kg/s"},
                {"Parameter": "Total Mass Flow", "Value": f"{base['m_dot_core']*(1+bpr_val):.3f}", "Unit": "kg/s"},
                {"Parameter": "Fuel Flow Rate", "Value": f"{base['m_dot_fuel']:.4f}", "Unit": "kg/s"},
                {"Parameter": "Fuel-Air Ratio (f)", "Value": f"{base['f']:.4f}", "Unit": "-"}
            ],
            "📐 Fan Geometry & Aerodynamics": [
                {"Parameter": "Fan Tip Diameter (D_tip)", "Value": f"{D_tip:.4f} ({D_tip*39.3701:.2f})", "Unit": "m (in)"},
                {"Parameter": "Fan Hub Diameter (D_hub)", "Value": f"{D_hub:.4f} ({D_hub*39.3701:.2f})", "Unit": "m (in)"},
                {"Parameter": "Fan Mean Diameter (D_mean)", "Value": f"{D_mean:.4f} ({D_mean*39.3701:.2f})", "Unit": "m (in)"},
                {"Parameter": "Fan Blade Height (h)", "Value": f"{h_blade:.4f} ({h_blade*39.3701:.2f})", "Unit": "m (in)"},
                {"Parameter": "Blade Aspect Ratio (AR)", "Value": f"{ar_val:.2f}", "Unit": "-"},
                {"Parameter": "Blade Chord", "Value": f"{c_blade:.4f}", "Unit": "m"}
            ],
            "⚡ Kinematics & Speed (RPM: {0})".format(f"{rpm_val:.0f}"): [
                {"Parameter": "Blade Tip Speed (U_tip)", "Value": f"{U_tip:.2f}", "Unit": "m/s"},
                {"Parameter": "Tip Relative Mach (M_rel,tip)", "Value": f"{M_rel_tip:.3f}", "Unit": "-"},
                {"Parameter": "Mean Relative Mach (M_rel,m)", "Value": f"{M_rel_mean:.3f}", "Unit": "-"}
            ],
            "🚀 Thrust Components": [
                {"Parameter": "Core Gross Thrust", "Value": f"{base['F_core']:.2f} ({base['F_core']/1000:.2f})", "Unit": "N (kN)"},
                {"Parameter": "Bypass Gross Thrust", "Value": f"{base['F_byp']:.2f} ({base['F_byp']/1000:.2f})", "Unit": "N (kN)"},
                {"Parameter": "Ram Drag", "Value": f"{base['F_ram']:.2f} ({base['F_ram']/1000:.2f})", "Unit": "N (kN)"},
                {"Parameter": "Net Thrust (F_net)", "Value": f"{base['F_net']:.2f} ({base['F_net']/1000:.2f})", "Unit": "N (kN)"}
            ],
            "📈 Engine Performance & Efficiencies": [
                {"Parameter": "Specific Thrust (Total Air)", "Value": f"{base['F_spec_total']:.4f}", "Unit": "N/(kg/s)"},
                {"Parameter": "TSFC", "Value": f"{base['TSFC']:.4f}", "Unit": "mg/N/s"},
                {"Parameter": "SFC", "Value": f"{base['SFC']:.4f}", "Unit": "kg/(kN·h)"},
                {"Parameter": "Thermal Efficiency", "Value": f"{base['eta_th']:.2f}", "Unit": "%"},
                {"Parameter": "Propulsive Efficiency", "Value": f"{base['eta_p']:.2f}", "Unit": "%"},
                {"Parameter": "Overall Efficiency", "Value": f"{base['eta_o']:.2f}", "Unit": "%"}
            ]
        }
        
        for category, rows in perf_categories.items():
            with st.expander(category, expanded=True):
                st.table(rows)

    # --- Tab 2: Station Profiles ---
    with tabProfiles:
        st.subheader("Station Profiles (Temperature & Pressure)")
        c_idx = [0, 1, 4, 5, 6, 7, 8, 9, 10]
        labels_core = [base['st_names'][i] for i in c_idx]
        b_idx = [0, 1, 2, 3]
        
        col_left, col_center, col_right = st.columns([1, 4, 1])
        with col_center:
            fig_prof, (axTemp, axPress) = plt.subplots(2, 1, figsize=(6, 4.5), constrained_layout=True, facecolor='none')
            
            axTemp.plot(range(len(c_idx)), base['Tt_K'][c_idx], color='#e11d48', marker='o', linewidth=1.5, label='Core T_t')
            axTemp.plot(range(len(c_idx)), base['T_K'][c_idx], color='#f97316', marker='s', linestyle='--', linewidth=1.2, label='Core T')
            axTemp.plot(range(len(b_idx)), base['Tt_K'][b_idx], color='#9333ea', marker='^', linestyle='-', linewidth=1.2, label='Bypass T_t,byp')
            axTemp.set_xticks(range(len(c_idx)))
            axTemp.set_xticklabels(labels_core, fontsize=7.5, color='#111827')
            axTemp.tick_params(axis='y', labelsize=7.5, colors='#111827')
            axTemp.grid(True, linestyle='--', alpha=0.3, color='#9ca3af')
            axTemp.set_title('Temperature Variation Across Engine Stations', fontsize=9.5, fontweight='bold', color='#111827')
            axTemp.set_ylabel('Temperature (K)', fontsize=7.5, color='#111827')
            axTemp.legend(fontsize=6.5, facecolor='#ffffff', edgecolor='#9ca3af', labelcolor='#111827')
            axTemp.set_facecolor('#ffffff')
            axTemp.set_xlim(-0.5, len(c_idx) - 0.5)
            
            axPress.plot(range(len(c_idx)), base['Pt_kPa'][c_idx], color='#0284c7', marker='o', linewidth=1.5, label='Core P_t')
            axPress.plot(range(len(c_idx)), base['P_kPa'][c_idx], color='#0284c7', marker='s', linestyle='--', linewidth=1.2, label='Core P')
            axPress.plot(range(len(b_idx)), base['Pt_kPa'][b_idx], color='#0d9488', marker='^', linestyle='-', linewidth=1.2, label='Bypass P_t,byp')
            axPress.set_xticks(range(len(c_idx)))
            axPress.set_xticklabels(labels_core, fontsize=7.5, color='#111827')
            axPress.tick_params(axis='y', labelsize=7.5, colors='#111827')
            axPress.grid(True, linestyle='--', alpha=0.3, color='#9ca3af')
            axPress.set_title('Pressure Variation Across Engine Stations', fontsize=9.5, fontweight='bold', color='#111827')
            axPress.set_ylabel('Pressure (kPa)', fontsize=7.5, color='#111827')
            axPress.legend(fontsize=6.5, facecolor='#ffffff', edgecolor='#9ca3af', labelcolor='#111827')
            axPress.set_facecolor('#ffffff')
            axPress.set_xlim(-0.5, len(c_idx) - 0.5)
            
            st.pyplot(fig_prof, use_container_width=True)

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
            
        col_left, col_center, col_right = st.columns([1, 4, 1])
        with col_center:
            fig_ts, axTS = plt.subplots(figsize=(6, 4.0), constrained_layout=True, facecolor='none')
            axTS.plot(s_core, T_core, color='#e11d48', marker='o', linewidth=1.5, label='Core Flow Path')
            axTS.plot(s_byp, T_byp, color='#2563eb', marker='s', linestyle='--', linewidth=1.3, label='Bypass Flow Path')
            axTS.grid(True, linestyle='--', alpha=0.3, color='#9ca3af')
            axTS.set_title('Temperature–Entropy (T–s) Diagram', fontsize=9.5, fontweight='bold', color='#111827')
            axTS.set_xlabel('Delta s (J/kg·K)', fontsize=7.5, fontweight='bold', color='#111827')
            axTS.set_ylabel('Total Temperature T_t (K)', fontsize=7.5, fontweight='bold', color='#111827')
            axTS.tick_params(axis='both', labelsize=7.5, colors='#111827')
            axTS.legend(fontsize=7, facecolor='#ffffff', edgecolor='#9ca3af', labelcolor='#111827')
            axTS.set_facecolor('#ffffff')
            axTS.set_xlim(s_core.min(), s_core.max())
            
            st.pyplot(fig_ts, use_container_width=True)

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
                
        fig_1d, axs = plt.subplots(2, 3, figsize=(11, 5), constrained_layout=True, facecolor='none')
        for ax_row in axs:
            for ax in ax_row:
                ax.set_facecolor('#ffffff')
                ax.set_xlim(s_range.min(), s_range.max())
                
        axs[0, 0].plot(s_range, res_F, color='#0284c7', linewidth=1.8); axs[0, 0].grid(True, alpha=0.3); axs[0, 0].set_title('Thrust (kN)', fontsize=9, fontweight='bold', color='#111827'); axs[0, 0].set_xlabel(x_lbl, fontsize=8, color='#111827'); axs[0, 0].tick_params(labelsize=7, colors='#111827')
        axs[0, 1].plot(s_range, res_TSFC, color='#dc2626', linewidth=1.8); axs[0, 1].grid(True, alpha=0.3); axs[0, 1].set_title('TSFC (mg/N/s)', fontsize=9, fontweight='bold', color='#111827'); axs[0, 1].set_xlabel(x_lbl, fontsize=8, color='#111827'); axs[0, 1].tick_params(labelsize=7, colors='#111827')
        axs[0, 2].plot(s_range, res_th, color='#16a34a', linewidth=1.8); axs[0, 2].grid(True, alpha=0.3); axs[0, 2].set_title('Thermal Eff. (%)', fontsize=9, fontweight='bold', color='#111827'); axs[0, 2].set_xlabel(x_lbl, fontsize=8, color='#111827'); axs[0, 2].tick_params(labelsize=7, colors='#111827')
        axs[1, 0].plot(s_range, res_p, color='#9333ea', linewidth=1.8); axs[1, 0].grid(True, alpha=0.3); axs[1, 0].set_title('Propulsive Eff. (%)', fontsize=9, fontweight='bold', color='#111827'); axs[1, 0].set_xlabel(x_lbl, fontsize=8, color='#111827'); axs[1, 0].tick_params(labelsize=7, colors='#111827')
        axs[1, 1].plot(s_range, res_o, color='#0d9488', linewidth=1.8); axs[1, 1].grid(True, alpha=0.3); axs[1, 1].set_title('Overall Eff. (%)', fontsize=9, fontweight='bold', color='#111827'); axs[1, 1].set_xlabel(x_lbl, fontsize=8, color='#111827'); axs[1, 1].tick_params(labelsize=7, colors='#111827')
        axs[1, 2].axis('off')
        st.pyplot(fig_1d)

    # --- Tab 5: 2D Contour Maps ---
    with tab2D:
        st.subheader("2D Performance Contour Maps (MATLAB Style Layout)")
        colX, colY = st.columns(2)
        items2D = ['TIT', '\\pi_f', '\\pi_{lpc}', 'BPR', '\\pi_{hpc}', 'Alt', '\\eta_f', '\\eta_{lpc}', '\\eta_{hpc}', '\\eta_{hpt}', '\\eta_{lpt}']
        
        with colX:
            varX = st.selectbox("X-Axis Parameter:", items2D, index=1)
        with colY:
            varY = st.selectbox("Y-Axis Parameter:", items2D, index=5)
            
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
                    
        fig_2d, axs_2d = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True, facecolor='none')
        plot_styled_contour(axs_2d[0, 0], X_mesh, Y_mesh, Z_F, lblX, lblY, 'Net Thrust (kN)')
        plot_styled_contour(axs_2d[0, 1], X_mesh, Y_mesh, Z_TSFC, lblX, lblY, 'TSFC (mg/N/s)')
        plot_styled_contour(axs_2d[1, 0], X_mesh, Y_mesh, Z_th, lblX, lblY, 'Thermal Efficiency (%)')
        plot_styled_contour(axs_2d[1, 1], X_mesh, Y_mesh, Z_o, lblX, lblY, 'Overall Efficiency (%)')
        
        st.pyplot(fig_2d)