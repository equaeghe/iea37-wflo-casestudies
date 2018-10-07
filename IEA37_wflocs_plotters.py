import matplotlib.pyplot as plt
import patches as pts  # https://stackoverflow.com/a/24568380


def layout_scatter(initial, final, radius):
    # initial layout
    pts.circles(initial.x, initial.y, 1, c='b')
    # final layout
    pts.circles(final.x, final.y, 1, c='r')
    # border
    border = plt.Circle((0, 0), radius, color='k', fill=False)
    plt.gcf().gca().add_artist(border)
    # appearance tweaks
    plt.axis('equal')
    plt.axis([-radius-1, radius+1, -radius-1, radius+1])
