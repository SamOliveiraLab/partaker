import sys
from matplotlib import pyplot as plt
from image_functions import remove_stage_jitter_MAE
from segmentation import segment_images
from track_view import save_tracks

### Specified Paths ###
source_path = './sample_data/XY8_Long_PHC/'
output_path = './aligned_data/XY8_Long_PHC/'

YFP_path = './sample_data/XY8_Long_YFP/'
YFP_output_path = './aligned_data/XY8_Long_YFP/'

Cherry_path = './sample_data/XY8_Long_Cherry/'
Cherry_output_path = './aligned_data/XY8_Long_Cherry/'

if sys.argv[1] == 'align':
    stage_MAE_scores = remove_stage_jitter_MAE(
        output_path,
        source_path,
        YFP_path,
        YFP_output_path,
        Cherry_path,
        Cherry_output_path,
        10000,
        -15,
        True,
        False
    )

# Run btrack here
elif sys.argv[1] == 'track':
    from tracking import track_cells

    segmentation = segment_images()

    tracks = track_cells(segmentation)

    save_tracks(segmentation, tracks)

"""
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock
from image_functions import remove_stage_jitter_MAE

from functools import partial
import tkinter as tk
from tkinter import filedialog

class AlignmentApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_path = ''
        self.output_path = ''
        self.YFP_path = ''
        self.YFP_output_path = ''
        self.Cherry_path = ''
        self.Cherry_output_path = ''
        self.progress_bar = None
        self.open_file_chooser = self._open_file_chooser_system

    def build(self):
        self.main_layout_box = BoxLayout(orientation='vertical')

        src_sel_layout = BoxLayout(orientation='horizontal')
        self.src_sel_layout = src_sel_layout
        self.main_layout_box.add_widget(src_sel_layout)

        src_sel_layout.add_widget(Label(text='Select Paths:'))

        source_button = Button(text='Source Path')
        source_button.bind(on_release=self.open_file_chooser)
        src_sel_layout.add_widget(source_button)

        YFP_button = Button(text='YFP Path')
        YFP_button.bind(on_release=self.open_file_chooser)
        src_sel_layout.add_widget(YFP_button)

        Cherry_button = Button(text='Cherry Path')
        Cherry_button.bind(on_release=self.open_file_chooser)
        src_sel_layout.add_widget(Cherry_button)

        cb_layout = BoxLayout(orientation='horizontal')
        cb_layout.add_widget(Label(text='Auto Create Output Paths:'))
        output_checkbox = CheckBox()
        output_checkbox.bind(active=self.toggle_output_buttons)
        cb_layout.add_widget(output_checkbox)
        self.main_layout_box.add_widget(cb_layout)

        align_button = Button(text='Align Images')
        align_button.bind(on_release=self.align_images)
        self.main_layout_box.add_widget(align_button)

        # Output paths buttons
        self.out_sel_layout = BoxLayout(orientation='horizontal')
        output_button = Button(text='Output Path')
        output_button.bind(on_release=self.open_file_chooser)
        YFP_output_button = Button(text='YFP Output Path')
        YFP_output_button.bind(on_release=self.open_file_chooser)
        Cherry_output_button = Button(text='Cherry Output Path')
        Cherry_output_button.bind(on_release=self.open_file_chooser)
        self.output_button = output_button
        self.YFP_output_button = YFP_output_button
        self.Cherry_output_button = Cherry_output_button
        self.out_sel_layout.add_widget(self.output_button)
        self.out_sel_layout.add_widget(self.YFP_output_button)
        self.out_sel_layout.add_widget(self.Cherry_output_button)

        return self.main_layout_box

    def toggle_output_buttons(self, checkbox, value):
        if value:
            self.main_layout_box.add_widget(self.out_sel_layout)
        else:
            self.main_layout_box.remove_widget(self.out_sel_layout)

        self.auto_create_output_paths = value

    def _open_file_chooser_system(self, button):
        root = tk.Tk()
        root.withdraw()
        selected_path = filedialog.askopenfilename()
        text_input = TextInput(text=selected_path, multiline=False)
        self.root.add_widget(text_input)
        if 'Source Path' in self.popup.title:
            self.source_path = selected_path
        elif 'Output Path' in self.popup.title:
            self.output_path = selected_path
        elif 'YFP Path' in self.popup.title:
            self.YFP_path = selected_path
        elif 'YFP Output Path' in self.popup.title:
            self.YFP_output_path = selected_path
        elif 'Cherry Path' in self.popup.title:
            self.Cherry_path = selected_path
        elif 'Cherry Output Path' in self.popup.title:
            self.Cherry_output_path = selected_path

    def _open_file_chooser(self, button):
        file_chooser = FileChooserListView()
        file_chooser.bind(selection=self.file_selected)
        popup = Popup(title='Select Path', content=file_chooser, size_hint=(0.9, 0.9))
        popup.open()

    def file_selected(self, file_chooser):
        selected_path = file_chooser.selection and file_chooser.selection[0] or ''
        text_input = TextInput(text=selected_path, multiline=False)
        self.popup.dismiss()
        self.popup = None
        self.root.add_widget(text_input)

        if 'Source Path' in self.popup.title:
            self.source_path = selected_path
        elif 'Output Path' in self.popup.title:
            self.output_path = selected_path
        elif 'YFP Path' in self.popup.title:
            self.YFP_path = selected_path
        elif 'YFP Output Path' in self.popup.title:
            self.YFP_output_path = selected_path
        elif 'Cherry Path' in self.popup.title:
            self.Cherry_path = selected_path
        elif 'Cherry Output Path' in self.popup.title:
            self.Cherry_output_path = selected_path

    def align_images(self, button):
        if not self.source_path or not self.output_path or not self.YFP_path or not self.YFP_output_path or not self.Cherry_path or not self.Cherry_output_path:
            self.show_error_popup('Please select all paths.')
            return

        self.progress_bar = ProgressBar(max=100)
        self.popup = Popup(title='Alignment Progress', content=self.progress_bar, size_hint=(0.3, 0.3))
        self.popup.open()

        Clock.schedule_once(partial(self.perform_alignment, self.source_path, self.output_path, self.YFP_path, self.YFP_output_path, self.Cherry_path, self.Cherry_output_path), 0.1)

    def perform_alignment(self, source_path, output_path, YFP_path, YFP_output_path, Cherry_path, Cherry_output_path, dt):
        stage_MAE_scores = remove_stage_jitter_MAE(
            output_path,
            source_path,
            YFP_path,
            YFP_output_path,
            Cherry_path,
            Cherry_output_path,
            10000,
            -15,
            True,
            False
        )

        self.progress_bar.value = 100
        self.popup.dismiss()
        self.popup = None

        self.show_info_popup('Alignment completed successfully.')

    def show_error_popup(self, message):
        popup = Popup(title='Error', content=Label(text=message), size_hint=(0.3, 0.3))
        popup.open()

    def show_info_popup(self, message):
        popup = Popup(title='Info', content=Label(text=message), size_hint=(0.3, 0.3))
        popup.open()

if __name__ == '__main__':

    AlignmentApp().run()


# YFP_path = './sample_data/XY8_Long_YFP/';
# YFP_output_path = './aligned_data/XY8_Long_YFP/';

# Cherry_path = './sample_data/XY8_Long_Cherry/';
# Cherry_output_path = './aligned_data/XY8_Long_Cherry/';

# stage_MAE_scores = remove_stage_jitter_MAE(
#     output_path,
#     source_path,
#     YFP_path,
#     YFP_output_path,
#     Cherry_path,
#     Cherry_output_path,
#     10000,
#     -15,
#     True,
#     False
# );

"""
