---
# try also 'default' to start simple
theme: default
# apply UnoCSS classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
colorSchema: light
---
<style>
.full-bg {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;        /* cobre o slide inteiro */
  z-index: -1;              /* fica atrás do conteúdo */
}
</style>

<img src="./background/90941.jpg" alt="background" class="full-bg" />

# Self-adaptive Loss Balanced Physics-Informed Neural Networks
<br>
<br>

## Eric Mochiutti  
<br>
<br>
<div class="footer-info">
Universidade de São Paulo  
<br>
<br>
<br>
Dezembro de 2025
</div>

---
transition: fade-out
---

# Para que serve pesos adaptativos no contexto de PINNs?

<v-click>

- **Balancear o Erro (Termos da Função de Custo):** Atuam como **multiplicadores** dos custos, ajustando dinamicamente a importância de cada componente da função de perda. Com o objetivo de evitar o <span v-mark.underline.red= "3">*training*</span> <span v-mark.underline.red= "3">*bottleneck*</span>.

</v-click>

<v-click>

<div class="flex justify-center mt-8">
  <img src="/images/PINNS.jpg" alt="Diagrama de Balanceamento de Loss em PINNs" class="rounded-2xl shadow-md w-3.1/5">
</div>

</v-click>

<style>

ul {
  font-size: 1.1rem;
  line-height: 1.7rem;
  margin-top: 1.8rem;
}

li {
  margin-bottom: 0.8rem;
  text-align: justify;
}
</style>

---
transition: fade-out
---

# Método Self-Adaptive Loss Balanced (lbPINNs)
<br>
<v-click>

### Ponto chave: Incerteza como Parâmetro Adaptativo
<br>
</v-click>

<v-click>

1.  **Problema Multitarefa:** A função de custo total de PINNs $(L_{total} = L_{PDE} + L_{BC} + ...)$ é vista como um problema de otimização multiobjetivo.

</v-click>

<v-click>

2.  **Base no MLE Gaussiano:** O método se inspira no aprendizado profundo para multitarefas, onde a importância de cada custo é determinada pela sua <span v-mark.underline.red= "3"> **incerteza**</span>.

</v-click>

<v-click>

3.  **Peso Dinâmico:** Se a incerteza é baixa, o modelo confia mais naquele termo, e seu peso aumenta.

</v-click>

<v-click>

<div class="flex justify-center mt-8">
  <img src="/images/artigo_ref.png" alt="Artigo Base" class="shadow-md w-3/6">
</div>

</v-click>

---
transition: fade-in
---

# Máxima Verossimilhança (MLE) para Pesos por Incerteza  

<v-click>

- Assumimos que o erro dessa tarefa vem de uma **distribuição gaussiana** com variância desconhecida.

</v-click>

<v-click>

$$
p\big(y \,\big|\, \hat{y}(x,t;\theta)\big)
= \mathcal{N}\!\left(\hat{y}(x,t;\theta),\, \sigma^2\right)
$$

- Modela a relação entre a saída real $(y)$ e a previsão da rede neural
$(\hat{y}(x,t;\theta))$. Ela assume que o erro de predição segue uma
distribuição Gaussiana de média dada pela aproximação da rede neural e variância $(\sigma^2)$.

</v-click>


<v-click>

- A lógica da Máxima Verossimilhança (MLE) é escolher os parâmetros
$(\theta)$ e a variância $(\sigma^2)$ que tornam os dados observados mais
prováveis sob o modelo Gaussiano. A modelagem final do artigo resulta em, após aplicar a minimização do log da gaussiana e transformar para termos exponenciais:

$$
L(s;\,\theta,\,N) =
\frac{1}{2}\exp(-s_{pde})\,L_{\mathrm{PDE}}(\theta)
\;+\;
\frac{1}{2}\exp(-s_{bc})\,L_{\mathrm{BC}}(\theta)
\;+
s_{pde} + s_{bc}
$$

</v-click>

---
transition: fade-in
---

# Treinamento PINN tradicional


```python
def train_standard(model, inputs_bc, targets_bc, inputs_pde, alpha, epochs=3000):
    print("\n--- Standard Training ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    # Modification: history is now a dictionary to store all loss components
    #history = {"loss": [], "loss_bc": [], "loss_pde": []}

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # 1. Boundary Condition Loss (Data-driven loss)
        loss_bc = loss_fn(model(inputs_bc), targets_bc)
        # 2. PDE Residual Loss (Physics-driven loss)
        loss_pde = torch.mean(calculate_physics_residual(model, inputs_pde) ** 2)

        # Weighted Total Loss (Fixed weights: 1-alpha and alpha)
        loss = (1 - alpha) * loss_bc + alpha * loss_pde

        loss.backward()
        optimizer.step()

    return history

```

---
transition: fade-in
---

# Treinamento PINN adaptativa

```python
def train_adaptive(model, inputs_bc, targets_bc, inputs_pde, epochs=3000):
    device = inputs_bc.device
    model.to(device)
    model.train()
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3)
    s_bc = torch.tensor(2.0, requires_grad=True, device=device) // {1|6,7,8}
    s_pde = torch.tensor(1.0, requires_grad=True, device=device) // {2|6,7,8}
    optimizer_w = optim.Adam([s_bc, s_pde], lr=1e-3) // {2|6,7,8}
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer_model.zero_grad()
        optimizer_w.zero_grad()

        loss_bc = loss_fn(model(inputs_bc), targets_bc)
        loss_pde = torch.mean(calculate_physics_residual(model, inputs_pde) ** 2)

        loss = (1 / 2 * torch.exp(-s_bc) * loss_bc+ 1 / 2 * torch.exp(-s_pde) * loss_pde+ s_bc+ s_pde)

        loss.backward()
        optimizer_model.step()
        optimizer_w.step()

    return history
```
---
transition: None
---

# Treinamento PINN adaptativa

```python{6,7,8}
def train_adaptive(model, inputs_bc, targets_bc, inputs_pde, epochs=3000):
    device = inputs_bc.device
    model.to(device)
    model.train()
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3)
    s_bc = torch.tensor(2.0, requires_grad=True, device=device) // {1|6,7,8}
    s_pde = torch.tensor(1.0, requires_grad=True, device=device) // {2|6,7,8}
    optimizer_w = optim.Adam([s_bc, s_pde], lr=1e-3) // {2|6,7,8}
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer_model.zero_grad()
        optimizer_w.zero_grad()

        loss_bc = loss_fn(model(inputs_bc), targets_bc)
        loss_pde = torch.mean(calculate_physics_residual(model, inputs_pde) ** 2)

        loss = (1 / 2 * torch.exp(-s_bc) * loss_bc+ 1 / 2 * torch.exp(-s_pde) * loss_pde+ s_bc+ s_pde)

        loss.backward()
        optimizer_model.step()
        optimizer_w.step()

    return history
```

---
transition: None
---

# Treinamento PINN adaptativa

```python{18}
def train_adaptive(model, inputs_bc, targets_bc, inputs_pde, epochs=3000):
    device = inputs_bc.device
    model.to(device)
    model.train()
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3)
    s_bc = torch.tensor(2.0, requires_grad=True, device=device) // {1|6,7,8}
    s_pde = torch.tensor(1.0, requires_grad=True, device=device) // {2|6,7,8}
    optimizer_w = optim.Adam([s_bc, s_pde], lr=1e-3) // {2|6,7,8}
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer_model.zero_grad()
        optimizer_w.zero_grad()

        loss_bc = loss_fn(model(inputs_bc), targets_bc)
        loss_pde = torch.mean(calculate_physics_residual(model, inputs_pde) ** 2)

        loss = (1 / 2 * torch.exp(-s_bc) * loss_bc+ 1 / 2 * torch.exp(-s_pde) * loss_pde+ s_bc+ s_pde)

        loss.backward()
        optimizer_model.step()
        optimizer_w.step()

    return history
```

---
transition: None
---

# Treinamento PINN adaptativa

```python{22}
def train_adaptive(model, inputs_bc, targets_bc, inputs_pde, epochs=3000):
    device = inputs_bc.device
    model.to(device)
    model.train()
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3)
    s_bc = torch.tensor(2.0, requires_grad=True, device=device) // {1|6,7,8}
    s_pde = torch.tensor(1.0, requires_grad=True, device=device) // {2|6,7,8}
    optimizer_w = optim.Adam([s_bc, s_pde], lr=1e-3) // {2|6,7,8}
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer_model.zero_grad()
        optimizer_w.zero_grad()

        loss_bc = loss_fn(model(inputs_bc), targets_bc)
        loss_pde = torch.mean(calculate_physics_residual(model, inputs_pde) ** 2)

        loss = (1 / 2 * torch.exp(-s_bc) * loss_bc+ 1 / 2 * torch.exp(-s_pde) * loss_pde+ s_bc+ s_pde)

        loss.backward()
        optimizer_model.step()
        optimizer_w.step()

    return history
```
---
transition: fade in
---
# Resultados

<div class="flex justify-center gap-6 mt-8">
  <img src="/images/exact_solution.png" alt="Imagem 10" class="shadow-md w-1/3">
  <img src="/images/standard_pinn_prediction.png" alt="Imagem 20" class="shadow-md w-1/3">
  <img src="/images/adaptive_pinn_prediction.png" alt="Imagem 30" class="shadow-md w-1/3">
</div>

---
transition: fade in
---

# Diminuição do erro

<div class="flex justify-center gap-6 mt-8">
  <img src="/images/absolute_error_standard_pinn.png" alt="Imagem 1" class="shadow-md w-1/2">
  <img src="/images/absolute_error_adaptive_pinn.png" alt="Imagem 2" class="shadow-md w-1/2">
</div>

---
transition: fade
---

# Treinamento relativamente estável

<!-- Linha de cima (2 imagens) -->
<div class="flex justify-center gap-6 mt-8">
  <img src="/images/training_history_standard_pinn.png" alt="Imagem 1" class="shadow-md w-1/2">
  <img src="/images/training_history_adaptive_pinn.png" alt="Imagem 2" class="shadow-md w-1/2">
</div>

<!-- Linha de baixo (1 imagem centralizada) -->
<div class="flex justify-center mt-6">
  <img src="/images/adaptive_weights_history.png" alt="Imagem 3" class="shadow-md w-0.9/2">
</div>

---
transition: fade in
---
# Infelizmente nunca é fácil

<div class="flex justify-center gap-6 mt-8">
  <img src="/images/exact_solution_2.png" alt="Imagem 10" class="shadow-md w-1/3">
  <img src="/images/standard_pinn_prediction_2.png" alt="Imagem 20" class="shadow-md w-1/3">
  <img src="/images/adaptive_pinn_prediction_2.png" alt="Imagem 30" class="shadow-md w-1/3">
</div>

---
transition: fade in
---

# Erro maior que da PINN tradicional, sensibilidade ao chute inicial


<div class="flex justify-center gap-6 mt-8">
  <img src="/images/absolute_error_standard_pinn_2.png" alt="Imagem 1" class="shadow-md w-1/2">
  <img src="/images/absolute_error_adaptive_pinn_2.png" alt="Imagem 2" class="shadow-md w-1/2">
</div>

---
transition: fade
---

# F.I.M

<!-- Linha de cima (2 imagens) -->
<div class="flex justify-center gap-6 mt-8">
  <img src="/images/training_history_standard_pinn_2.png" alt="Imagem 1" class="shadow-md w-1/2">
  <img src="/images/training_history_adaptive_pinn_2.png" alt="Imagem 2" class="shadow-md w-1/2">
</div>

<!-- Linha de baixo (1 imagem centralizada) -->
<div class="flex justify-center mt-6">
  <img src="/images/adaptive_weights_history_2.png" alt="Imagem 3" class="shadow-md w-0.9/2">
</div>