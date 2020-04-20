import matplotlib.pyplot as plt
import numpy as np
image_names = ["astronaut.png", "chelsea.png"]

orb_times = [59.264960099999996, 37.415723299999996]
orb_c_times = [81.36152279999999, 50.35558889999999]

if __name__ == '__main__':

    x = np.arange(len(image_names))  # the label locations
    width = 0.45  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, orb_times, width, label='ORB')
    rects2 = ax.bar(x + width / 2, orb_c_times, width, label='ORB-c')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time (s)')
    ax.set_title('Speed comparison between ORB and ORB-c')
    ax.set_xticks(x)
    plt.xticks(rotation=35)
    ax.set_xticklabels(image_names)
    ax.legend()
    fig.tight_layout()

    plt.show()