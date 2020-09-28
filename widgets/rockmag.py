import ipywidgets as widgets
from ipywidgets import *#AppLayout, Button, Layout, GridspecLayout, interactive, interact


def widget():
    table_style = {'description_width': 'initial'}
    table_layout = {'width': '150px', 'min_width': '150px', 'height': '28px', 'min_height': '28px'}
    row_layout = {'width': '200px', 'min_width': '200px'}

    table_header_0_widget = Button(
        description='',
        disabled=True,
        button_style='',
        tooltip='',
        icon='',
        layout=row_layout,
        style=table_style
    )

    table_header_1_widget = Button(description='header 1',
                                   disabled=True,
                                   button_style='',
                                   tooltip='',
                                   icon='',
                                   # layout=header_layout,
                                   # style=table_style
                                   layout=table_layout
                                   )
    table_header_2_widget = Button(description='header 2',
                                   disabled=True,
                                   button_style='',
                                   tooltip='',
                                   icon='',
                                   # layout=header_layout,
                                   # style=table_style
                                   layout=table_layout
                                   )
    table_header_3_widget = Button(description='header 3',
                                   disabled=True,
                                   button_style='',
                                   tooltip='',
                                   icon='',
                                   # layout=header_layout,
                                   # style=table_style
                                   layout=table_layout
                                   )

    row_1_0_widget = Button(
        description='row2 looooooooooooong:',
        disabled=True,
        button_style='',
        tooltip='',
        icon='',
        layout=row_layout,
        style=table_style
    )
    row_1_1_widget = BoundedFloatText(
        value=70.0,
        min=30.0,
        max=300.0,
        step=1.0,
        layout=table_layout,
        style=table_style,
    )
    row_1_2_widget = BoundedFloatText(
        value=80.0,
        min=30.0,
        max=300.0,
        step=1.0,
        description='',
        layout=table_layout,
        style=table_style
    )
    row_1_3_widget = BoundedFloatText(
        value=90.0,
        min=30.0,
        max=300.0,
        step=1.0,
        description='',
        layout=table_layout,
        style=table_style
    )
    row_2_0_widget = Button(
        description='row3:',
        disabled=True,
        button_style='',
        tooltip='',
        icon='',
        layout=row_layout,
        style=table_style
    )
    row_2_1_widget = BoundedFloatText(
        value=20.0,
        min=1.0,
        max=100.0,
        step=1.0,
        layout=table_layout,
        style=table_style
    )
    row_2_2_widget = BoundedFloatText(
        value=30.0,
        min=1.0,
        max=100.0,
        step=1.0,
        description='',
        layout=table_layout,
        style=table_style
    )
    row_2_3_widget = BoundedFloatText(
        value=40.0,
        min=1.0,
        max=100.0,
        step=1.0,
        description='',
        layout=table_layout,
        style=table_style
    )

    hbox1 = HBox([table_header_0_widget, table_header_1_widget, table_header_2_widget, table_header_3_widget])
    hbox2 = HBox([row_1_0_widget, row_1_1_widget, row_1_2_widget, row_1_3_widget])
    hbox3 = HBox([row_2_0_widget, row_2_1_widget, row_2_2_widget, row_2_3_widget])
    ui = VBox([hbox1, hbox2, hbox3])


    def func(p1, p2, p3, p4, p5, p6):
        print(p1, p2, p3, p4, p5, p6)


    w = interactive_output(func,
                           {
                               "p1": row_1_1_widget,
                               "p2": row_1_2_widget,
                               "p3": row_1_3_widget,
                               "p4": row_2_1_widget,
                               "p5": row_2_2_widget,
                               "p6": row_2_3_widget,
                           })

    display(ui, w)