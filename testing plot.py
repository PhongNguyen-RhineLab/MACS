import matplotlib.pyplot as plt
import numpy as np

# Set up the figure for 3 side-by-side plots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Common data
n = np.linspace(0, 10, 100)
n0 = 3  # The point where the condition starts holding


def wavy_line(x, slope, intercept, amplitude=0.2, freq=2):
    """Generates a line with a slight sine wave for a 'hand-drawn' feel"""
    return slope * x + intercept + amplitude * np.sin(freq * x)


# --- Graph (a): Theta Notation ---
ax = axs[0]
c2g = wavy_line(n, 1.2, 0.5, 0.3, 1.5)
f = wavy_line(n, 0.8, 0.8, 0.5, 2.5)
c1g = wavy_line(n, 0.5, 0.2, 0.2, 1.2)

ax.plot(n, c2g, 'k', label='$c_2g(n)$')
ax.plot(n, f, 'k', label='$f(n)$')
ax.plot(n, c1g, 'k', label='$c_1g(n)$')

# Annotations
ax.text(9, c2g[-1] + 0.2, '$c_2g(n)$')
ax.text(9, f[-1] + 0.2, '$f(n)$')
ax.text(9, c1g[-1] + 0.2, '$c_1g(n)$')
ax.set_title('$f(n) = \\Theta(g(n))$\n(a)', y=-0.25)

# --- Graph (b): Big-O Notation ---
ax = axs[1]
cg = wavy_line(n, 1.5, 1.0, 0.4, 1.2)
# f starts high then dips below cg after n0
f = 0.8 * n + 2 + 2 * np.sin(1.5 * n) * np.exp(-0.3 * n)

ax.plot(n, cg, 'k')
ax.plot(n, f, 'k')

ax.text(9, cg[-1] + 0.2, '$cg(n)$')
ax.text(9, f[-1] + 0.2, '$f(n)$')
ax.set_title('$f(n) = O(g(n))$\n(b)', y=-0.25)

# --- Graph (c): Big-Omega Notation ---
ax = axs[2]
cg = wavy_line(n, 0.5, 1.5, 0.2, 1.0)
# f starts low then stays above cg after n0
f = 1.2 * n + 0.5 + 1.5 * np.cos(2 * n) * np.exp(-0.2 * n)

ax.plot(n, cg, 'k')
ax.plot(n, f, 'k')

ax.text(9, cg[-1] + 0.2, '$cg(n)$')
ax.text(9, f[-1] + 0.2, '$f(n)$')
ax.set_title('$f(n) = \\Omega(g(n))$\n(c)', y=-0.25)

# General formatting for all subplots
for ax in axs:
    # Drawing the n0 vertical dashed line
    ax.axvline(x=n0, ymin=0, ymax=0.5, color='gray', linestyle='--')
    ax.text(n0, -0.5, '$n_0$', ha='center')

    # Axis styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('$n$', loc='right')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 15)

    # Create the L-shape axis
    ax.plot(0, 0, "k>", ms=10, clip_on=False, transform=ax.get_yaxis_transform())
    ax.plot(0, 0, "k^", ms=10, clip_on=False, transform=ax.get_xaxis_transform())

plt.tight_layout()
plt.show()