#!/usr/bin/env python3
"""
GUI Manager Module - Creates and manages graphical user interfaces.
Actually creates windows instead of just saying it will.
"""

import logging
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime

class GUIManager:
    """
    Manages GUI creation and display for PHOENIX.
    """

    def __init__(self, system_control=None):
        """
        Initialize the GUI Manager.

        Args:
            system_control: SystemControl module for executing commands
        """
        self.system_control = system_control
        self.logger = logging.getLogger("PHOENIX.GUI")

        # Track active GUIs
        self.active_guis = {}
        self.gui_scripts_dir = Path('./data/gui_scripts')
        self.gui_scripts_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("GUI Manager initialized")

    def create_schedule_gui(self, schedule_data: Dict) -> Dict[str, Any]:
        """
        Create a GUI for displaying schedule.

        Args:
            schedule_data: Schedule information to display

        Returns:
            Result of GUI creation
        """
        result = {
            'success': False,
            'message': '',
            'script_path': None,
            'process': None
        }

        try:
            # Generate Python script for the GUI
            script_content = self._generate_schedule_gui_script(schedule_data)

            # Save script to file
            script_path = self.gui_scripts_dir / f"schedule_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            with open(script_path, 'w') as f:
                f.write(script_content)

            # Execute the script
            if self.system_control:
                # Run GUI in background using subprocess directly
                import subprocess
                try:
                    process = subprocess.Popen(
                        [sys.executable, str(script_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True
                    )

                    result['success'] = True
                    result['script_path'] = str(script_path)
                    result['message'] = "Schedule GUI window opened successfully"
                    result['process'] = process
                    self.logger.info(f"GUI created: {script_path}")

                    # Track active GUI
                    self.active_guis[str(script_path)] = {
                        'type': 'schedule',
                        'created': datetime.now().isoformat(),
                        'data': schedule_data,
                        'pid': process.pid
                    }
                except Exception as e:
                    result['message'] = f"Failed to open GUI: {e}"
            else:
                # Fallback: try direct subprocess
                import subprocess
                process = subprocess.Popen([sys.executable, str(script_path)],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)

                result['success'] = True
                result['process'] = process
                result['script_path'] = str(script_path)
                result['message'] = "Schedule GUI window opened"

        except Exception as e:
            self.logger.error(f"Failed to create GUI: {e}")
            result['message'] = f"Error creating GUI: {e}"

        return result

    def _generate_schedule_gui_script(self, schedule_data: Dict) -> str:
        """
        Generate a Python script for the schedule GUI.

        Args:
            schedule_data: Schedule data to display

        Returns:
            Python script content
        """
        # Convert schedule data to JSON for embedding
        schedule_json = json.dumps(schedule_data, indent=2)

        script = f'''#!/usr/bin/env python3
"""
Auto-generated Schedule GUI
Created by PHOENIX GUI Manager
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import json
from datetime import datetime, timedelta
import calendar

class ScheduleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tennis Schedule - PHOENIX")
        self.root.geometry("900x600")

        # Schedule data
        self.schedule_data = {schedule_json}

        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="🎾 Tennis Schedule",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Week view tab
        self.week_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.week_frame, text="Week View")
        self.create_week_view()

        # Day view tab
        self.day_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.day_frame, text="Today")
        self.create_day_view()

        # Pending tasks tab
        self.pending_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pending_frame, text="Pending Tasks")
        self.create_pending_view()

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Refresh", command=self.refresh).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Add Lesson", command=self.add_lesson).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=root.quit).pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E))

    def create_week_view(self):
        """Create the week view."""
        # Create treeview for week schedule
        columns = ('Time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
        self.week_tree = ttk.Treeview(self.week_frame, columns=columns, show='headings', height=20)

        # Define headings
        for col in columns:
            self.week_tree.heading(col, text=col)
            self.week_tree.column(col, width=100)

        self.week_tree.column('Time', width=80)

        # Add scrollbars
        vsb = ttk.Scrollbar(self.week_frame, orient="vertical", command=self.week_tree.yview)
        hsb = ttk.Scrollbar(self.week_frame, orient="horizontal", command=self.week_tree.xview)
        self.week_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid layout
        self.week_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.week_frame.columnconfigure(0, weight=1)
        self.week_frame.rowconfigure(0, weight=1)

        # Populate with schedule data
        self.populate_week_view()

    def populate_week_view(self):
        """Populate the week view with schedule data."""
        # Clear existing items
        for item in self.week_tree.get_children():
            self.week_tree.delete(item)

        # Create time slots
        for hour in range(7, 21):  # 7 AM to 9 PM
            time_slot = f"{{hour:02d}}:00"
            row_data = [time_slot]

            # Check each day for lessons at this time
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                if day in self.schedule_data:
                    day_lessons = self.schedule_data[day]
                    lesson_text = ""
                    for lesson in day_lessons:
                        start_hour = int(lesson.get('start_time', '00:00').split(':')[0])
                        if start_hour == hour:
                            student = lesson.get('student_name', 'Student')
                            lesson_text = f"{{student[:10]}}"
                            break
                    row_data.append(lesson_text)
                else:
                    row_data.append("")

            self.week_tree.insert('', tk.END, values=row_data)

    def create_day_view(self):
        """Create the day view."""
        # Get today's schedule
        today = calendar.day_name[datetime.now().weekday()]

        # Title for today
        day_title = ttk.Label(self.day_frame, text=f"Today: {{today}}",
                             font=('Arial', 14, 'bold'))
        day_title.pack(pady=10)

        # Create frame for lessons
        lessons_frame = ttk.Frame(self.day_frame)
        lessons_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        if today in self.schedule_data and self.schedule_data[today]:
            for i, lesson in enumerate(self.schedule_data[today]):
                lesson_frame = ttk.LabelFrame(lessons_frame,
                                            text=f"{{lesson.get('start_time', '')}} - {{lesson.get('end_time', '')}}",
                                            padding="10")
                lesson_frame.pack(fill=tk.X, pady=5)

                ttk.Label(lesson_frame, text=f"Student: {{lesson.get('student_name', 'Unknown')}}").pack(anchor=tk.W)

                if lesson.get('court_booked'):
                    status_text = "✅ Court Booked"
                else:
                    status_text = "⚠️ Need to book court at Tennis 13"

                ttk.Label(lesson_frame, text=status_text).pack(anchor=tk.W)
        else:
            ttk.Label(lessons_frame, text="No lessons scheduled for today").pack(pady=20)

    def create_pending_view(self):
        """Create the pending tasks view."""
        # Title
        pending_title = ttk.Label(self.pending_frame, text="Pending Tasks",
                                font=('Arial', 14, 'bold'))
        pending_title.pack(pady=10)

        # Text widget for pending items
        self.pending_text = scrolledtext.ScrolledText(self.pending_frame, width=60, height=15)
        self.pending_text.pack(padx=10, pady=10)

        # Add pending court bookings
        pending_items = []
        if 'pending_court_booking' in self.schedule_data:
            for lesson in self.schedule_data.get('pending_court_booking', []):
                pending_items.append(f"⚠️ Book court for {{lesson.get('date', '')}} at {{lesson.get('start_time', '')}}")

        if pending_items:
            self.pending_text.insert(tk.END, "\\n".join(pending_items))
        else:
            self.pending_text.insert(tk.END, "No pending tasks!")

        self.pending_text.config(state=tk.DISABLED)

    def refresh(self):
        """Refresh the display."""
        self.status_var.set("Refreshing...")
        self.populate_week_view()
        self.status_var.set("Refreshed at " + datetime.now().strftime("%H:%M:%S"))

    def add_lesson(self):
        """Open dialog to add a new lesson."""
        # Create new window for adding lesson
        add_window = tk.Toplevel(self.root)
        add_window.title("Add New Lesson")
        add_window.geometry("400x300")

        # Form fields
        ttk.Label(add_window, text="Student Name:").grid(row=0, column=0, padx=10, pady=5)
        name_entry = ttk.Entry(add_window, width=30)
        name_entry.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(add_window, text="Date:").grid(row=1, column=0, padx=10, pady=5)
        date_entry = ttk.Entry(add_window, width=30)
        date_entry.grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(add_window, text="Start Time:").grid(row=2, column=0, padx=10, pady=5)
        start_entry = ttk.Entry(add_window, width=30)
        start_entry.grid(row=2, column=1, padx=10, pady=5)

        ttk.Label(add_window, text="End Time:").grid(row=3, column=0, padx=10, pady=5)
        end_entry = ttk.Entry(add_window, width=30)
        end_entry.grid(row=3, column=1, padx=10, pady=5)

        def save_lesson():
            self.status_var.set("Lesson added (demo only - not saved)")
            add_window.destroy()

        ttk.Button(add_window, text="Save", command=save_lesson).grid(row=4, column=0, pady=20)
        ttk.Button(add_window, text="Cancel", command=add_window.destroy).grid(row=4, column=1, pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = ScheduleGUI(root)
    root.mainloop()
'''

        return script

    def create_simple_window(self, title: str, content: str) -> Dict[str, Any]:
        """
        Create a simple text window.

        Args:
            title: Window title
            content: Content to display

        Returns:
            Result of window creation
        """
        result = {
            'success': False,
            'message': '',
            'script_path': None
        }

        try:
            # Generate simple GUI script
            script = f'''#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, scrolledtext

root = tk.Tk()
root.title("{title}")
root.geometry("600x400")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

text = scrolledtext.ScrolledText(frame, width=70, height=20)
text.grid(row=0, column=0)
text.insert(tk.END, """{content}""")
text.config(state=tk.DISABLED)

ttk.Button(frame, text="Close", command=root.quit).grid(row=1, column=0, pady=10)

root.mainloop()
'''

            # Save and execute
            script_path = self.gui_scripts_dir / f"window_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            with open(script_path, 'w') as f:
                f.write(script)

            # Execute
            if self.system_control:
                success, _, stderr = self.system_control.run_command(
                    f"python3 {script_path}",
                    background=True
                )
                if success:
                    result['success'] = True
                    result['script_path'] = str(script_path)
                    result['message'] = f"Window '{title}' opened"

        except Exception as e:
            self.logger.error(f"Failed to create window: {e}")
            result['message'] = f"Error: {e}"

        return result

    def get_capabilities(self) -> Dict[str, bool]:
        """Return actual capabilities of this module."""
        return {
            'create_windows': True,
            'schedule_gui': True,
            'simple_text_window': True,
            'interactive_forms': True,
            'close_windows': False,  # Not yet implemented
            'real_time_updates': False  # Not yet implemented
        }