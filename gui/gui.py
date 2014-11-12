import Tkinter as tk
#import Tkinter.ttk as ttk
import thread
import time
import os

class App:

    def __init__(self):

        self.running = True

        root = tk.Tk()
        self.root = root

        root.wm_title("PAMPRO")

        sources_frame = tk.Frame(root, padx=5, pady=5)#, bg="darkgrey")
        progress_frame = tk.Frame(root, padx=5, pady=5)#, bg="darkgrey")
        analysis_frame = tk.Frame(root, padx=5, pady=5)#, bg="darkgrey")
        metafile_frame = tk.Frame(root, padx=5, pady=5)#, bg="darkgrey")

        sources_frame.grid(row=0, column=0, sticky="NW")
        #progress_frame.grid(row=1, column=0, sticky="N")
        analysis_frame.grid(row=1, column=0, sticky="NW")
        metafile_frame.grid(row=2, column=0, sticky="NW")

        self.num_sources = 0
        self.num_sources_value = tk.StringVar()
        self.num_sources_value.set("(0)")

        font_properties = ("Helvetica", 10)
        button_font_properties = ("Helvetica", 8)
        header_font_properties = ("Helvetica", 10, "bold")
        button_properties = {"bg":"lightgrey", "fg":"black", "bd":0, "activeforeground":"black", "activebackground":"lightblue", "relief":"flat", "font":button_font_properties, "height":1, "padx":5, "pady":1}
        entry_properties = {"relief":"flat", "bd":1, "font":font_properties}
        listbox_properties = {"relief":"flat", "bd":1, "font":font_properties}
        label_properties = {"font":font_properties, "pady":1}
        header_label_properties = {"font":header_font_properties}


        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Sources frame

        self.scrollbar = tk.Scrollbar(sources_frame, orient=tk.VERTICAL)

        self.source_label = tk.Label(sources_frame, text="Select source data:", **header_label_properties)
        self.source_entry = tk.Entry(sources_frame, **entry_properties)
        self.load_button = tk.Button(sources_frame, text="Load", command=self.click_load, **button_properties)
        self.browse_button = tk.Button(sources_frame, text="Browse", command=self.click_browse, **button_properties)
        self.files_listbox = tk.Listbox(sources_frame, width=40,yscrollcommand=self.scrollbar.set, **listbox_properties)
        self.source_value_label = tk.Label(sources_frame, textvariable=self.num_sources_value, **label_properties)

        self.source_label.grid(row=0, column=0, padx=5, pady=0, sticky="W")
        self.source_value_label.grid(row=0, column=1, padx=5, pady=0, sticky="W")
        self.files_listbox.grid(row=2, column=0, padx=5, pady=5, columnspan=3, sticky="W")
        self.source_entry.grid(row=1, column=0, padx=5, pady=5, sticky="W")
        self.load_button.grid(row=1, column=2, padx=5, pady=5, sticky="W")
        self.browse_button.grid(row=1, column=1, padx=5, pady=5, sticky="W")



        self.scrollbar.config(command=self.files_listbox.yview)
        self.scrollbar.grid(row=2,column=1)
        #self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        #self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Progress frame

        self.process_button = tk.Button(progress_frame, text="Process", command=self.click_process, **button_properties)
        self.progress_label = tk.Label(progress_frame, text="Progress:", **label_properties)
        self.progress_value_label = tk.Label(progress_frame, text="0%", **label_properties)

        self.progress_label.grid(row=0,column=0, padx=5, pady=5)
        self.progress_value_label.grid(row=0,column=1, padx=5, pady=5)
        self.process_button.grid(row=0, column=2, padx=5, pady=5)


        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Analysis frame

        self.analysis_label = tk.Label(analysis_frame, text="Select analysis script:", **header_label_properties)
        self.analysis_entry = tk.Entry(analysis_frame, **entry_properties)
        self.analysis_button = tk.Button(analysis_frame, text="Browse", command=self.click_browse, **button_properties)

        self.analysis_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="W")
        self.analysis_entry.grid(row=1, column=0, padx=5, pady=5, sticky="W")
        self.analysis_button.grid(row=1, column=1, padx=5, pady=5, sticky="W")

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Meta file frame

        self.metafile_label = tk.Label(metafile_frame, text="Select meta file:", **header_label_properties)
        self.metafile_entry = tk.Entry(metafile_frame, **entry_properties)
        self.metafile_button = tk.Button(metafile_frame, text="Browse", command=self.click_browse, **button_properties)

        self.metafile_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="W")
        self.metafile_entry.grid(row=1, column=0, padx=5, pady=5, sticky="W")
        self.metafile_button.grid(row=1, column=1, padx=5, pady=5, sticky="W")


        thread.start_new_thread(self.loop, ())

    def loop(self):

        self.root.mainloop()

        self.running = False
   

    def click_load(self):

        try:
            for f in os.listdir(self.source_entry.get()):
                if f.endswith(".txt"):
                    self.files_listbox.insert(tk.END, self.source_entry.get() + "/" + f)

                    self.num_sources += 1

        except:
            pass#

        self.num_sources_value.set("(" + str(self.num_sources) + ")")

    def click_browse(self):

        pass

    def click_process(self):

        pass

    def click_length(self):

        pass



app = App()

while (app.running):
    print "Test"
    time.sleep(1)