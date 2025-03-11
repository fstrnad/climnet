from imp import reload
import climnet.utils.spatial_utils as sput
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import climnet.plots as cplt
import climnet.utils.time_utils as tu
from tqdm import tqdm


def movie_map(ds,
              dmap,
              frame_rate=400,
              intpol=True,
              plot_type="contourf",
              central_longitude=0,
              vmin=0,
              vmax=1,
              cmap=None,
              bar=True,
              projection="EqualEarth",
              label=None,
              title=None,
              significant_mask=False,
              kde_plot=False,
              lon_range=None,
              lat_range=None,
              **kwargs):
    reload(cplt)
    im = cplt.create_multi_map_plot(nrows=1,
                                    ncols=1,
                                    ds=ds,
                                    projection=projection,
                                    figsize=(11, 8),
                                    title=title,
                                    **kwargs
                                    )

    frames = dmap.time
    idx = 0


    def animate(tp_frame, kde=kde_plot):
        im['ax'].cla()
        tp = tp_frame
        da_tp = dmap.sel(time=tp)
        if kde:
            Z_kde = sput.get_kde_map(
                ds,
                ds.flatten_array(da_tp, time=False,
                                 check=False),
                nn_points=5)
            da_tp = Z_kde
        # bar = True if idx == 0 else False
        bar = True
        im_plot = cplt.plot_map(ds, da_tp,
                                ax=im['ax'],
                                plot_type=plot_type,
                                intpol=intpol,
                                significant_mask=significant_mask,
                                central_longitude=central_longitude,
                                cmap=cmap,
                                vmin=vmin,
                                vmax=vmax,
                                bar=bar,
                                label=label,
                                **kwargs)
        tp_date = tu.get_ym_date(tp)
        cplt.text_box(ax=im['ax'], text=f'{tp_date}')

        if lon_range is not None and lat_range is not None:
            cplt.plot_rectangle(
                ax=im["ax"], lon_range=lon_range, lat_range=lat_range, color="tab:red", lw=3
            )

        # idx += 1

    ani = animation.FuncAnimation(plt.gcf(), animate, frames=tqdm(frames), interval=frame_rate,
                                  init_func=None, blit=False, )

    return ani
