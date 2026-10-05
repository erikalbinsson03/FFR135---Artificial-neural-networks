import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


M_values = [1, 2, 4, 8]
NU_MAX = 10 ** 5
P_0 = 10
K = 10
ETA = 0.005


N_RUNS = 10
BURN_IN = 10 ** 4
N_SAMPLES = 10 ** 6


data_patterns = np.array([[-1, -1, -1], [ 1, -1,  1], [-1,  1,  1], [ 1,  1, -1]])
all_patterns = np.array([[-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1], [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1]])


P_data = np.ones(4) / 4


all_DKL = {}
for M in M_values:
    all_DKL[M] = []


for M in M_values:

    for run in range(N_RUNS):
        W = np.random.randn(M, 3)
        theta_v = np.zeros(3)
        theta_h = np.zeros(M)
        progress_step = max(1, NU_MAX // 10)

        for nu in range(NU_MAX):
            delta_W = np.zeros((M, 3))
            delta_theta_v = np.zeros(3)
            delta_theta_h = np.zeros(M)

            for mu in range(P_0):
                index = np.random.randint(4)
                v_0 = data_patterns[index].copy()

                b_h0 = W @ v_0 - theta_h
                tanh_b_h0 = np.tanh(b_h0)
                p_h0 = (1 + tanh_b_h0) / 2

                h = np.where(np.random.rand(M) < p_h0, 1, -1)
                v = v_0.copy()

                for k in range(K):
                    b_v = W.T @ h - theta_v
                    p_v = (1 + np.tanh(b_v)) / 2
                    v = np.where(np.random.rand(3) < p_v, 1, -1)

                    b_h = W @ v - theta_h
                    p_h = (1 + np.tanh(b_h)) / 2
                    h = np.where(np.random.rand(M) < p_h, 1, -1)

                tanh_b_hk = np.tanh(b_h)
                delta_W += ETA * (np.outer(tanh_b_h0, v_0) - np.outer(tanh_b_hk, v))
                delta_theta_v += -ETA * (v_0 - v)
                delta_theta_h += -ETA * (tanh_b_h0 - tanh_b_hk)

            W += delta_W
            theta_v += delta_theta_v
            theta_h += delta_theta_h

            if (nu + 1) % progress_step == 0 or nu == NU_MAX - 1:
                progress = 100 * (nu + 1) / NU_MAX
                print(
                    f"M={M}, run {run + 1}/{N_RUNS}: "
                    f"training approximately {progress:.0f}% complete",
                    flush=True,
                )

        v = np.random.choice([-1, 1], size=3)

        for step in range(BURN_IN):
            b_h = W @ v - theta_h
            p_h = (1 + np.tanh(b_h)) / 2
            h = np.where(np.random.rand(M) < p_h, 1, -1)

            b_v = W.T @ h - theta_v
            p_v = (1 + np.tanh(b_v)) / 2
            v = np.where(np.random.rand(3) < p_v, 1, -1)

        counts = np.zeros(len(all_patterns))

        for step in range(N_SAMPLES):
            b_h = W @ v - theta_h
            p_h = (1 + np.tanh(b_h)) / 2
            h = np.where(np.random.rand(M) < p_h, 1, -1)

            b_v = W.T @ h - theta_v
            p_v = (1 + np.tanh(b_v)) / 2
            v = np.where(np.random.rand(3) < p_v, 1, -1)

            for i in range(len(all_patterns)):
                if np.array_equal(v, all_patterns[i]):
                    counts[i] += 1
                    break

        P_B = counts / N_SAMPLES

        D_KL = 0
        for i in range(4):
            pattern = data_patterns[i]

            for j in range(len(all_patterns)):
                if np.array_equal(pattern, all_patterns[j]):
                    P_model = P_B[j]

                    if P_model == 0:
                        D_KL = np.inf
                    else:
                        D_KL += P_data[i] * np.log(P_data[i] / P_model)
                    break

        all_DKL[M].append(D_KL)


best_DKL = {}
for M in M_values:
    best_DKL[M] = np.min(all_DKL[M])


results = pd.DataFrame({
    "M": M_values,
    "Best D_KL": [best_DKL[M] for M in M_values]
})

print("\nBest results:")
print(results)


plt.figure(figsize=(8, 5))

for M in M_values:
    x = np.ones(N_RUNS) * M
    plt.scatter(x, all_DKL[M], label=f"M = {M}")

plt.xlabel("Number of hidden neurons M")
plt.ylabel(r"$D_{KL}$")
plt.title(r"$D_{KL}$ for 10 independent runs")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(8, 5))

plt.plot(M_values, [best_DKL[M] for M in M_values], "o-")

plt.xlabel("Number of hidden neurons M")
plt.ylabel(r"Best $D_{KL}$")
plt.title(r"Best $D_{KL}$ vs number of hidden neurons")
plt.grid()
plt.show()


N = 3
theory_DKL = []

for M in M_values:
    if M < 2 ** (N - 1) - 1:
        L = int(np.floor(np.log2(M + 1)))
        bound = np.log(2) * (N - L - (M + 1) / (2 ** L))
    else:
        bound = 0
    theory_DKL.append(bound)


plt.figure(figsize=(8, 5))

plt.plot(M_values, [best_DKL[M] for M in M_values], "o-", label="Best numerical result")

plt.plot(M_values, theory_DKL,"s--", label="Theory, Eq. (4.40)")

plt.xlabel("Number of hidden neurons M")
plt.ylabel(r"$D_{KL}$")
plt.title(r"Numerical $D_{KL}$ vs. theoretical bound")
plt.legend()
plt.grid()
plt.show()
