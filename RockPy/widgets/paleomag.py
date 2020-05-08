import os
import pandas as pd

from scipy import stats
from io import StringIO, BytesIO
import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipywidgets import AppLayout, Button, Layout, GridspecLayout, interactive, interact
from mpl_toolkits.axisartist.axislines import SubplotZero

from IPython.display import clear_output, display

from RockPy.ftypes.cif import Cif
from RockPy.tools.plotting import *

import matplotlib as mpl

mpl.rcParams['toolbar'] = 'None'


def paleomag():
    # for debug ( std_out)
    output = widgets.Output()
    plt.ioff()
    plt.close('all')

    centered_layout = widgets.Layout(display='flex',
                                     justify_content='center',
                                     align_items='center'
                                     )
    ### Site level info
    site_grid = GridspecLayout(1, 5)

    site_lat = widgets.FloatText(description='Lat:', width='10px')
    site__lat = widgets.Label('˚N')

    site_long = widgets.FloatText(description='Long:')
    site__long = widgets.Label('˚E')

    site_dec = widgets.FloatText(description='Mag. Decl.:')
    site__dec = widgets.Label('˚')

    site_grid[0, 0] = widgets.HBox([site_lat, site__lat])
    site_grid[0, 1] = widgets.HBox([site_long, site__long])
    site_grid[0, 2] = widgets.HBox([site_dec, site__dec])

    header = site_grid

    ### Sample level infos
    angle_layout = Layout(width='250px')

    sample_core_dec = widgets.FloatText(description='Core dec:', value=0, layout=angle_layout)
    sample__core_dec = widgets.Label('˚')

    sample_core_inc = widgets.FloatText(description='inc:', value=0, layout=angle_layout)
    sample__core_inc = widgets.Label('˚')

    sample_strat_dec = widgets.FloatText(description='Strat dec:', value=0, layout=angle_layout)
    sample__strat_dec = widgets.Label('˚')

    sample_strat_inc = widgets.FloatText(description='inc:', value=0, layout=angle_layout)
    sample__strat_inc = widgets.Label('˚')

    reset_button = widgets.Button(description='reset', layout=Layout(width='70px'))

    sample_grid = widgets.HBox(
        [sample_core_dec, sample__core_dec, sample_core_inc, sample__core_inc, sample_strat_dec, sample__strat_dec,
         sample_strat_inc, sample__strat_inc, reset_button],
        layout=Layout(float='left'))

    footer = sample_grid
    footer.layout = centered_layout

    ### SAMPLE data/list

    sample_data = {}
    fit_parameter = {None: {'center_label': [], 'min': [], 'max': [], 'r2': [], 'slope': [], 'intercept': []}}

    global_sample_list = widgets.Select(
        options=[],
        description='',
        layout=Layout(width='auto')
    )

    def _sample_data_change_selection(change):
        with output:
            clear_output()

            s = change['new']

            if s is None:
                return
            # print(sample_data[s]['cif'].header['core_strike'].values)
            # try:
            sample_core_dec.value = sample_data[s]['cif'].header['core_strike'].values[0]
            sample_core_inc.value = sample_data[s]['cif'].header['core_dip'].values[0]
            sample_strat_dec.value = sample_data[s]['cif'].header['bedding_strike'].values[0]
            sample_strat_inc.value = sample_data[s]['cif'].header['bedding_dip'].values[0]
            # except:
            #     print(f'something went wrong assigning footer values of {s}')
            #     print(sample_data[s]['cif'].header)

    global_sample_list.observe(_sample_data_change_selection, names='value')

    upload = widgets.FileUpload(
        accept='',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
        multiple=True,  # True to accept multiple files upload else False
        layout=Layout(align_self='center'))

    ### PLOTS
    ## STEREONET
    stereonet = plt.figure()
    stereo_ax = plt.subplot(111, projection='polar', label='stereo_ax')
    stereonet.add_subplot(stereo_ax)
    setup_stereonet(grid=True, ax=stereo_ax)

    ortho = plt.figure()
    ortho_ax = plt.subplot(111, label='otrtho_ax')
    ortho.add_subplot(ortho_ax)

    ortho_ax.set_aspect('equal')

    def _redraw_figures_info(s):

        stereo_ax.clear()
        plot_equal(sample_data[s]['cif'].geo_xyz, ax=stereo_ax)
        stereo_ax.set_title(s)
        stereonet.canvas.draw()
        stereonet.canvas.flush_events()

        ortho_ax.clear()

        ortho_ax.spines['left'].set_position('center')
        ortho_ax.spines['right'].set_color('none')
        ortho_ax.spines['bottom'].set_position('center')
        ortho_ax.spines['top'].set_color('none')
        ortho_ax.spines['left'].set_smart_bounds(True)
        ortho_ax.spines['bottom'].set_smart_bounds(True)
        ortho_ax.xaxis.set_ticks_position('bottom')
        ortho_ax.yaxis.set_ticks_position('left')
        # for direction in ["xzero", "yzero"]:
        #     # adds X and Y-axis from the origin
        #     ortho_ax.axis[direction].set_visible(True)
        # 
        # for direction in ["left", "right", "bottom", "top"]:
        #     # hides borders
        #     ortho_ax.axis[direction].set_visible(False)

        ortho_ax.plot(sample_data[s]['cif'].geo_xyz['y'], sample_data[s]['cif'].geo_xyz['x'], marker='o')
        ortho_ax.plot(sample_data[s]['cif'].geo_xyz['y'], - sample_data[s]['cif'].geo_xyz['z'], marker='s')
        ortho.canvas.draw()
        ortho.canvas.flush_events()

    def _update_sample(change):
        with output:
            s = change['new']
            if s is None:
                return
            _redraw_figures_info(s)
            _redraw_figures_info(s)

    """ BACK - NEXT buttons """
    prev_button = widgets.Button(description='back')
    next_button = widgets.Button(description='next')

    def _back_button_on_click(change):
        with output:
            global_sample_list.value = global_sample_list.options[global_sample_list.index - 1]

    def _next_button_on_click(change):
        with output:
            try:
                global_sample_list.value = global_sample_list.options[global_sample_list.index + 1]
            except IndexError:
                global_sample_list.value = global_sample_list.options[0]

    prev_button.on_click(_back_button_on_click)
    next_button.on_click(_next_button_on_click)

    back_next_buttons = widgets.HBox([prev_button, next_button], layout=centered_layout)

    def _on_upload(change):
        new_d = change['new']
        with output:
            clear_output()
            for s in new_d:
                fit_parameter.setdefault(s, fit_parameter[None])
                try:
                    new_d[s]['cif'] = Cif(BytesIO(new_d[s]['content']))
                except:
                    print(f'something went wrong while importing {s}')

        sample_data.update(change['new'])
        global_sample_list.options = list(sample_data.keys())
        global_sample_list.notify_change(
            {'name': 'value', 'old': global_sample_list.value, 'new': global_sample_list.options[-1],
             'owner': global_sample_list, 'type': 'change'})

    global_sample_list.observe(_update_sample, names='value')

    upload.observe(_on_upload, names='value')
    uploader = widgets.VBox([back_next_buttons,
                             widgets.Label('Samples'), global_sample_list, upload],
                            )

    ## changin the footer values

    def _on_angle_change(change):
        with output:
            clear_output()
            s = global_sample_list.value
            # print('change in footer', change)
            # print('geo dec, inc: ', sample_core_dec.value, sample_core_inc.value)
            # print('strat dec, inc: ', sample_strat_dec.value, sample_strat_inc.value)
            # print(sample_data[s]['cif'].header)

            if change['owner'].description in ('Core dec:', 'inc:'):
                if sample_data[s]['cif'].header['core_strike'].values[0] == sample_core_dec.value and \
                        sample_data[s]['cif'].header['core_dip'].values[0] == sample_core_inc.value:
                    return

                sample_data[s]['cif'].reset_geo(dip=sample_core_inc.value, strike=sample_core_dec.value)
                _redraw_figures_info(s)
                return

            if change['owner'].description in ('Strat dec:', 'inc:'):
                if sample_data[s]['cif'].header['bedding_strike'].values[0] == sample_strat_dec.value and \
                        sample_data[s]['cif'].header['bedding_dip'].values[0] == sample_strat_inc.value:
                    return

                sample_data[s]['cif'].reset_strat(dip=sample_strat_inc.value, strike=sample_strat_dec.value)
                _redraw_figures_info(s)
                return

    sample_core_dec.observe(_on_angle_change, names='value')
    sample_core_inc.observe(_on_angle_change, names='value')
    sample_strat_dec.observe(_on_angle_change, names='value')
    sample_strat_inc.observe(_on_angle_change, names='value')

    def _on_reset(change):
        with output:
            clear_output()
            s = global_sample_list.value
            sample_data[s]['cif'].reset_strat()
            sample_data[s]['cif'].reset_geo()

            sample_core_dec.value = sample_data[s]['cif'].header['core_strike'].values[0]
            sample_core_inc.value = sample_data[s]['cif'].header['core_dip'].values[0]
            sample_strat_dec.value = sample_data[s]['cif'].header['bedding_strike'].values[0]
            sample_strat_inc.value = sample_data[s]['cif'].header['bedding_dip'].values[0]
            _redraw_figures_info(s)

    reset_button.on_click(_on_reset)

    """ fitting tab definintions """

    ### PLOTS
    ## STEREONET
    # stereonet_fit = plt.figure()
    # stereo_ax_fit = plt.subplot(111, projection='polar', center_label='stereo_fit')
    # stereonet_fit.add_subplot(stereo_ax_fit)
    # setup_stereonet(grid=True, ax=stereo_ax_fit)
    #
    # ortho_fit = plt.figure()
    # ortho_ax_fit = plt.subplot(111, center_label='ortho_fit')
    # ortho_fit.add_subplot(ortho_ax_fit)
    #
    # ortho_ax_fit.set_aspect('equal')

    # def _redraw_figures_fit(s):
    #
    #     stereo_ax_fit.clear()
    #     plot_equal(sample_data[s]['cif'].geo_xyz, ax=stereo_ax_fit)
    #     stereo_ax_fit.set_title(s)
    #     stereonet_fit.canvas.draw()
    #     stereonet_fit.canvas.flush_events()
    #
    #     ortho_ax_fit.clear()
    #     for direction in ["xzero", "yzero"]:
    #         # adds X and Y-axis from the origin
    #         ortho_ax_fit.axis[direction].set_visible(True)
    #
    #     for direction in ["left", "right", "bottom", "top"]:
    #         # hides borders
    #         ortho_ax_fit.axis[direction].set_visible(False)
    #
    #     ortho_ax_fit.plot(sample_data[s]['cif'].geo_xyz['y'], sample_data[s]['cif'].geo_xyz['x'], marker='o')
    #     ortho_ax_fit.plot(sample_data[s]['cif'].geo_xyz['y'], - sample_data[s]['cif'].geo_xyz['z'], marker='s')
    #     ortho_fit.canvas.draw()
    #     ortho_fit.canvas.flush_events()

    fit_column_width = Layout(flex='4 1 0%', width='auto')

    global_fit_list = widgets.Select(
        options=[],
        description='',
        layout=Layout(width='auto')
    )
    global_min_fit_parameters = widgets.Select(
        options=[],
        description='',
        layout=fit_column_width
    )
    global_max_fit_parameters = widgets.Select(
        options=[],
        description='',
        layout=fit_column_width
    )
    global_goodnes_fit = widgets.Select(
        options=[],
        description='',
        layout=fit_column_width
    )

    global_fit_list_box = widgets.HBox([global_fit_list, global_min_fit_parameters,
                                        global_max_fit_parameters, global_goodnes_fit],
                                       layout=centered_layout)

    global_fit_list_header = widgets.HBox([widgets.Button(description='center_label', layout=Layout(width='auto')),
                                           widgets.Button(description='min', layout=fit_column_width),
                                           widgets.Button(description='max', layout=fit_column_width),
                                           widgets.Button(description='r$^2$', layout=fit_column_width)],
                                          layout=Layout(main_size='800px', justify_content='space-between'))
    # link the tables
    l1 = widgets.link((global_fit_list, 'index'), (global_min_fit_parameters, 'index'))
    l2 = widgets.link((global_fit_list, 'index'), (global_max_fit_parameters, 'index'))
    l3 = widgets.link((global_fit_list, 'index'), (global_goodnes_fit, 'index'))

    def _add_remove_fit_on_click(c):
        with output:
            print(c)
            print(fit_parameter)
            if c.description == '+ fit':
                # get selected sample
                s = global_sample_list.value
                print(s)
                if s is None:
                    print('add data first')

                # get number of fits for sample
                n = len(fit_parameter[s]['center_label'])
                add_fit_item(n, s)

                print(fit_parameter[s]['center_label'])
                global_fit_list.options = fit_parameter[s]['center_label']
                global_min_fit_parameters.options = fit_parameter[s]['min']
                global_max_fit_parameters.options = fit_parameter[s]['max']
                global_goodnes_fit.options = fit_parameter[s]['r2']

    def add_fit_item(n, s):
        # add a new fit item
        fit_parameter[s]['center_label'].append(n + 1)
        fit_parameter[s]['min'].append(sample_data[s]['cif'].data.index.min())
        fit_parameter[s]['max'].append(sample_data[s]['cif'].data.index.max())
        fit_parameter[s]['r2'].append(np.inf)
        fit_parameter[s]['slope'].append(np.nan)
        fit_parameter[s]['intercept'].append(np.nan)

    add_fit_button = widgets.Button(description='+ fit')
    remove_fit_button = widgets.Button(description='- fit')

    add_fit_button.on_click(_add_remove_fit_on_click)
    remove_fit_button.on_click(_add_remove_fit_on_click)

    fit_buttons = widgets.HBox([add_fit_button, remove_fit_button], layout=centered_layout)
    #
    sample_select_fit = widgets.VBox([back_next_buttons,
                                      widgets.Label('Samples'), global_sample_list,
                                      global_fit_list_header,
                                      global_fit_list_box,
                                      fit_buttons,
                                      ], )

    fit_parameter_slider = widgets.SelectionRangeSlider(description=f"included fit parameter",
                                                        options=np.linspace(0, 1),
                                                        readout=True,
                                                        readout_format='.1f',
                                                        value=(0, 1))

    fitting_footer = widgets.HBox([fit_parameter_slider], layout=centered_layout)

    def _on_fit_list_select(c):
        with output:
            print(c)

    global_fit_list.observe(_on_fit_list_select, names='index')
    # global_min_fit_parameters
    # global_max_fit_parameters
    # global_goodnes_fit
    #
    # APP LAYOUT
    data_tab = AppLayout(header=header,
                         left_sidebar=uploader,
                         center=widgets.VBox(
                             [stereonet.canvas,
                              back_next_buttons],
                         ),
                         right_sidebar=ortho.canvas,
                         footer=footer,
                         pane_widths=[2, 3, 3],
                         pane_heights=[1, 5, '60px'])

    fit_tab = AppLayout(header=header,
                        left_sidebar=sample_select_fit,
                        center=widgets.VBox(
                            [  # stereonet_fit.canvas,
                                back_next_buttons],
                        ),
                        right_sidebar=None,  # ortho_fit.canvas,
                        footer=fitting_footer,
                        pane_widths=[2, 3, 3],
                        pane_heights=[1, 5, '60px'])

    children = [data_tab, fit_tab, widgets.Text()]

    tab = widgets.Tab()
    tab.children = children

    tab.set_title(title='sample infos', index=0)
    tab.set_title(title='fitting', index=1)
    tab.set_title(title='tbd', index=2)

    display(tab, output)
