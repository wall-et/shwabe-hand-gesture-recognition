import tkinter as tk

class Gui_View:
    def __init__(self, functs_setup):
        self.funcs = functs_setup
        master = tk.Tk()
        self.master = master
        self.set_window_init()


    def set_window_init(self):
        self.master.title("Shwabe")
        self.draw_panel()

    def draw_panel(self):
        self.master_panel = tk.Frame(self.master, borderwidth=2, bg='white')
        self.master_panel.grid(padx=20, pady=4, sticky=tk.W + tk.E + tk.N + tk.S)

        self.file_button = tk.Button(self.master_panel, text="Run Shwabe", command=self.funcs['main_loop_start'],
                                     width=15, height=1, bg='white', font=("Arial", 11))
        self.file_button.config(font=("Arial", 11))
        self.file_button.grid()

        self.file_button = tk.Button(self.master_panel, text="Pause Shwabe", command=self.funcs['main_loop_stop'],
                                     width=15, height=1, bg='white', font=("Arial", 11))
        self.file_button.config(font=("Arial", 11))
        self.file_button.grid()