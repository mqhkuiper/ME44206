import numpy as np
import matplotlib.pyplot as plt

path_colors = ['r', 'b', 'g', 'k', 'y', 'm', 'c']

def draw_arrows(mx, my, color, arrow_spacing=2, mutation_scale=25):
    for i in range(len(mx) - 1):
        # Calculate the length of the segment
        x0, y0 = mx[i], my[i]
        x1, y1 = mx[i + 1], my[i + 1]
        segment_length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Number of arrows to place on the segment
        num_arrows = int(segment_length / arrow_spacing)
        for k in range(1, num_arrows + 1):
            # Interpolate the position of the arrow
            alpha = k / (num_arrows + 1)
            x_arrow = x0 + alpha * (x1 - x0)
            y_arrow = y0 + alpha * (y1 - y0)

            # Add an arrow at the interpolated position
            plt.annotate(
                '',  # Empty string for text
                xy=(x_arrow, y_arrow),  # Arrowhead location
                xytext=(x_arrow - (x1 - x0) * arrow_spacing / segment_length,
                        y_arrow - (y1 - y0) * arrow_spacing / segment_length),  # Arrow tail location
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=color,
                    lw=1,  # Line width for the arrow shaft
                    mutation_scale=mutation_scale  # Size of the arrowhead
                )
            )