import numpy as np

def get_alpha_from_lr(lr, min_alpha=0.3, max_alpha=1.0, lr_range=None):
    """Calculate alpha transparency based on the base learning rate."""
    if lr_range and lr_range[0] == lr_range[1]:  # Single learning rate case
        return max_alpha
    return min_alpha + (max_alpha - min_alpha) * (lr - lr_range[0]) / (lr_range[1] - lr_range[0])

def percentage_of_epoch(output, field, num_epochs):
    """Calculate the percentage of epochs for a given field."""
    total_iterations = len(output[field])
    percentages = [i / total_iterations * num_epochs for i in range(total_iterations)]
    return percentages

def plot_data(ax, outputs, field, ylabel, colormap, linestylemap, lr_ranges, alpha_func, zorder_func=None, wallclock=False):
    """Generalized function to plot data."""
    plotted_methods = set()
    for output in outputs:
        name, lr = output['name'].split('-lr-')
        lr = float(lr)
        alpha = alpha_func(lr, lr_range=lr_ranges[name])

        label = None
        if name not in plotted_methods:
            if lr_ranges[name][0] == lr_ranges[name][1]:  # Single learning rate
                label = f"{name} lr={lr_ranges[name][0]:.4f}"
            else:  # Range of learning rates
                label = f"{name} lr in [{lr_ranges[name][0]:.4f}, {lr_ranges[name][1]:.4f}]"

        zorder = zorder_func(name) if zorder_func else 1

        if wallclock:
            assert len(output["step_times"]) % len(output[field]) == 0
            step_factor = len(output["step_times"]) // len(output[field])
            step_times = np.array(output["step_times"])
            step_times = np.sum(step_times.reshape((len(output[field]), step_factor)), axis=1)
            xs = np.cumsum(step_times)
        else:
            xs = len(output[field])

        ax.plot(xs,
                output[field],
                label=label,
                color=colormap[name],
                linewidth=2,
                linestyle=linestylemap[name],
                alpha=alpha,
                zorder=zorder)
        plotted_methods.add(name)

    xlabel = "Seconds" if wallclock else "Epochs"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)