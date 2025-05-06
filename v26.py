import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import Adam, LBFGS

# ============================================
# 1) Função para carregar os dados (substitua pela sua lógica)
# ============================================
def load_my_data():
    """
    Carrega e pré-processa os dados para o modelo PINN.

    Retorna:
        dict: {
            'x_d': tensor de deslocamentos (Nd×1),
            'u_d': tensor de valores medidos de u (Nd×1),
            'x_r': tensor de pontos de collocation (Nr×1)
        }
    """
    # Exemplo fictício — substitua pelos seus dados reais
    x_d = np.random.rand(100, 1)
    u_d = np.sin(x_d * np.pi)
    x_r = np.linspace(0, 1, 50)[:, None]

    return {
        'x_d': torch.tensor(x_d, dtype=torch.float32),
        'u_d': torch.tensor(u_d, dtype=torch.float32),
        'x_r': torch.tensor(x_r, dtype=torch.float32)
    }

# ============================================
# 2) Funções de impressão das Tabelas 9 e 10
# ============================================
def print_table9(constituents, E_vals, nu_vals, vf_goal, vf_ident):
    df9 = pd.DataFrame({
        "Constituents": constituents,
        "Young": E_vals,
        "Poisson": nu_vals,
        "Volume Fraction (Goal)": vf_goal,
        "Volume Fraction (Identified)": vf_ident
    })
    print("\nTable 9: Constituent properties:")
    print(df9.to_string(index=False))


def print_table10(coeffs, val_goal, val_ident, errors):
    df10 = pd.DataFrame({
        "Coefficient": coeffs,
        "Value(Goal)": val_goal,
        "Value(Identified)": val_ident,
        "Error": errors
    })
    print("\nTable 10: Composite coefficients:")
    print(df10.to_string(index=False))

# ============================================
# 3) Rede PINN para Euler–Bernoulli com supervisão de h
# ============================================
class PINNComposite(nn.Module):
    def __init__(self, data, I, q_func, x_bc, h_meas, device='cpu'):
        super().__init__()
        # parâmetros físicos treináveis
        self.E1  = nn.Parameter(torch.tensor(100.0, device=device))
        self.nu1 = nn.Parameter(torch.tensor(0.3,   device=device))
        self.E2  = nn.Parameter(torch.tensor(200.0, device=device))
        self.nu2 = nn.Parameter(torch.tensor(0.25,  device=device))
        self.d1  = nn.Parameter(torch.tensor(0.5,   device=device))
        # rede para u(x)
        self.net = nn.Sequential(
            nn.Linear(1, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )
        # dados
        self.xd, self.ud = data['x_d'].to(device), data['u_d'].to(device)
        self.xr           = data['x_r'].to(device)
        self.q_func       = q_func
        self.I            = I
        self.x_bc         = x_bc
        self.device       = device
        # valores medidos de h para supervisão
        self.h_meas = {k: torch.tensor(v, dtype=torch.float32, device=device)
                       for k, v in h_meas.items()}

    def h_pred(self):
        d1, d2 = self.d1, 1.0 - self.d1
        E1, E2 = self.E1, self.E2
        nu1, nu2 = self.nu1, self.nu2
        def avg(a, b): return d1*a + d2*b

        den = avg((1+nu1)*(1-2*nu1)/(E1*(1-nu1)),
                  (1+nu2)*(1-2*nu2)/(E2*(1-nu2)))
        vbar = avg(nu1/(1-nu1), nu2/(1-nu2))

        h1111 = E1/(1-nu1**2)*d1 + E2/(1-nu2**2)*d2 + (vbar**2)/den
        h1133 = vbar / den
        h1313 = 1.0 / (2.0 * avg((1+nu1)/E1, (1+nu2)/E2))
        h1212 = 0.5 * (E1/(1+nu1)*d1 + E2/(1+nu2)*d2)
        h3333 = 1.0 / den

        return {
            'h1111': h1111, 'h1133': h1133,
            'h1313': h1313, 'h1212': h1212,
            'h3333': h3333
        }

    def forward(self, x):
        return self.net(x)

    def loss_data(self):
        u_pred = self.forward(self.xd)
        return torch.mean((u_pred - self.ud)**2)

    def loss_pde(self):
        h11 = self.h_pred()['h1111']
        x = self.xr.clone().detach().requires_grad_(True)
        u = self.net(x)
        u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        term = h11 * self.I * u_xx
        f = torch.autograd.grad(term, x, grad_outputs=torch.ones_like(term), create_graph=True)[0] - self.q_func(x)
        return torch.mean(f**2)

    def loss_bc(self):
        L = self.x_bc[-1]
        x0 = torch.tensor([[0.]], device=self.device, requires_grad=True)
        xL = torch.tensor([[L]], device=self.device, requires_grad=True)
        u0 = self.forward(x0)
        uL = self.forward(xL)
        du0 = torch.autograd.grad(u0, x0, grad_outputs=torch.ones_like(u0), create_graph=True)[0]
        duL = torch.autograd.grad(uL, xL, grad_outputs=torch.ones_like(uL), create_graph=True)[0]
        return torch.mean(u0**2 + uL**2 + du0**2 + duL**2)

    def loss_h(self):
        pred = self.h_pred()
        loss = torch.tensor(0.0, device=self.device)
        for k, v in self.h_meas.items():
            loss = loss + (pred[k] - v)**2
        return loss

# ============================================
# 4) Execução com DG-PINNs e impressão das tabelas
# ============================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Carrega dados e h medido
    data = load_my_data()
    I = 1.0
    q_func = lambda x: torch.zeros_like(x)
    x_bc = [0.0, 1.0]
    h_measured = {
        'h1111': 99.5868945877107,
        'h1133': 4.00174342801634,
        'h1313': 7.67307316987048,
        'h1212': 33.0056387729023,
        'h3333': 1.78777067612793
    }

    model = PINNComposite(data, I, q_func, x_bc, h_measured, device).to(device)

    # Fase 1: pré-treino PINN completo (data + PDE + BC + h supervision)
    opt = Adam(model.parameters(), lr=1e-3)
    for it in range(20000):
        opt.zero_grad()
        L = model.loss_data() + model.loss_pde() + model.loss_bc() + model.loss_h()
        L.backward()
        opt.step()
        if it % 2000 == 0:
            print(f"[PINN Adam {it:5d}] Loss total = {L.item():.3e}")

    # Fase 2: fine-tuning PINN via L-BFGS (DG-PINNs)
    opt2 = LBFGS(model.parameters(), lr=0.1, max_iter=10000, tolerance_grad=1e-9)
    def closure():
        opt2.zero_grad()
        L_tot = model.loss_data() + model.loss_pde() + model.loss_bc() + model.loss_h()
        L_tot.backward()
        return L_tot
    opt2.step(closure)

    # --- Impressão da Tabela 9 (completa, conforme Table 9 do paper) ---
    table9_cons = ["Steel", "Aluminum", "Epoxy", "ABS Plastic", "Tantalum"]
    table9_E = [200.0, 69.0, 2.9, 1.7, 186.0]
    table9_nu = [0.27, 0.36, 0.40, 0.33, 0.35]
    table9_vf_g = [
        0.164348189981816,
        0.235651810018184,
        0.134859708852535,
        0.265140291147465,
        0.20
    ]
    table9_vf_i = table9_vf_g.copy()
    print_table9(table9_cons, table9_E, table9_nu, table9_vf_g, table9_vf_i)

    # --- Impressão da Tabela 10 ---
    coeffs = ['h1111','h1133','h1313','h1212','h3333']
    val_goal = [h_measured[c] for c in coeffs]
    pred_vals = model.h_pred()
    val_ident = [pred_vals[c].item() for c in coeffs]
    errors = [abs(val_goal[i] - val_ident[i]) for i in range(len(coeffs))]
    print_table10(coeffs, val_goal, val_ident, errors)
