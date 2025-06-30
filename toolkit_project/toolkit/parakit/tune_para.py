#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter Tuning Module

A robust GUI-based parameter adjustment tool for machine learning and scientific computing.
Provides a user-friendly interface for modifying argparse parameters with automatic saving
and validation capabilities.

Author: Dongming Wang
Email: dongming.wang@email.ucr.edu
Version: 1.0.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, Toplevel, filedialog
import argparse
import json
import os
import threading
import time
from typing import Optional, Dict, Any, Union, Callable
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterTuner:
    """
    A GUI-based parameter tuner for argparse ArgumentParser objects.
    
    Features:
    - Interactive GUI for parameter adjustment
    - Auto-save functionality
    - Inactivity auto-save and close (10 seconds)
    - Configurable save paths
    - Parameter validation
    - Real-time status updates
    """
    
    def __init__(
        self,
        parser: argparse.ArgumentParser,
        save_delay: int = 30000,
        save_path: Optional[str] = None,
        auto_save_enabled: bool = True,
        validation_callbacks: Optional[Dict[str, Callable]] = None,
        inactivity_timeout: int = 10  # 10 seconds inactivity timeout
    ):
        """
        Initialize the ParameterTuner.
        
        Args:
            parser: ArgumentParser object to tune
            save_delay: Auto-save delay in milliseconds (default: 30 seconds)
            save_path: Directory to save parameters (default: current working directory)
            auto_save_enabled: Whether to enable auto-save
            validation_callbacks: Custom validation functions for parameters
            inactivity_timeout: Seconds of inactivity before auto-save and close
        """
        self.parser = parser
        self.save_delay = save_delay
        self.auto_save_enabled = auto_save_enabled
        self.validation_callbacks = validation_callbacks or {}
        self.inactivity_timeout = inactivity_timeout
        
        # Set default save path to current working directory
        if save_path is None:
            self.save_path = Path.cwd()
        else:
            self.save_path = Path(save_path)
        
        # Ensure save directory exists
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # GUI state
        self.root = None
        self.entries = {}
        self.status_label = None
        self.save_path_var = None
        self.auto_save_var = None
        self._auto_save_timer = None
        self._inactivity_timer = None
        self._is_saving = False
        
        # Extract parameter information
        self.args_defaults = self._extract_parameter_info()
        
        # Bind events for inactivity tracking
        self._last_activity = time.time()
    
    def _extract_parameter_info(self):
        """Extract parameter information from the parser."""
        return {
            action.dest: action 
            for action in self.parser._actions 
            if action.dest != 'help'
        }
    
    def _validate_parameter(self, key: str, value: str) -> tuple[bool, Any, str]:
        """
        Validate a parameter value.
        
        Args:
            key: Parameter name
            value: String value to validate
            
        Returns:
            Tuple of (is_valid, converted_value, error_message)
        """
        try:
            # Get the parameter action
            action = self.args_defaults[key]
            
            # Apply custom validation if available
            if key in self.validation_callbacks:
                is_valid, converted_value, error_msg = self.validation_callbacks[key](value)
                if not is_valid:
                    return False, None, error_msg
            
            # Type conversion
            if action.type:
                converted_value = action.type(value)
            else:
                converted_value = value
            
            # Check choices if specified
            if action.choices and converted_value not in action.choices:
                return False, None, f"Value must be one of: {action.choices}"
            
            return True, converted_value, ""
            
        except (ValueError, TypeError) as e:
            return False, None, str(e)
    
    def _show_message(self, title: str, message: str, message_type: str = "info", auto_close: bool = True):
        """Show a message dialog with optional auto-close."""
        if message_type == "error":
            messagebox.showerror(title, message)
        elif message_type == "warning":
            messagebox.showwarning(title, message)
        else:
            if auto_close:
                # Create auto-closing message window
                msg_window = Toplevel(self.root)
                msg_window.title(title)
                msg_window.geometry("400x100")
                msg_window.resizable(False, False)
                
                # Center the window
                msg_window.transient(self.root)
                msg_window.grab_set()
                
                label = ttk.Label(msg_window, text=message, padding=20, wraplength=350)
                label.pack(expand=True, fill="both")
                
                # Auto-close after 3 seconds
                msg_window.after(3000, msg_window.destroy)
            else:
                messagebox.showinfo(title, message)
    
    @classmethod
    def get_default_save_path(cls) -> Path:
        """Get the default save path (current working directory)."""
        return Path.cwd()
    
    def _save_parameters(self) -> bool:
        """Save current parameter values to file."""
        if self._is_saving:
            return False
        
        self._is_saving = True
        
        try:
            # Collect current parameter values
            parameters = {}
            validation_errors = []
            
            for key, entry in self.entries.items():
                value_str = entry.get().strip()
                is_valid, converted_value, error_msg = self._validate_parameter(key, value_str)
                
                if is_valid:
                    parameters[key] = converted_value
                else:
                    validation_errors.append(f"{key}: {error_msg}")
            
            if validation_errors:
                error_message = "Validation errors:\n" + "\n".join(validation_errors)
                self._show_message("Validation Error", error_message, "error")
                return False
            
            # Generate filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"parameters_{timestamp}.json"
            filepath = self.save_path / filename
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(parameters, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Parameters saved to: {filepath}")
            
            # Show success message with file info
            success_message = f"Parameters saved successfully!\n\nFile: {filename}\nLocation: {self.save_path}"
            self._show_message("Save Successful", success_message)
            
            # Update status
            self.status_label.config(text=f"Saved: {filename}", font=("Times New Roman", 8))
            
            # If this was triggered by OK button, close the window
            if hasattr(self, '_ok_clicked') and self._ok_clicked:
                self.root.quit()
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to save parameters: {str(e)}"
            logger.error(error_msg)
            self._show_message("Save Error", error_msg, "error")
            return False
        finally:
            self._is_saving = False
    
    def _save_and_close(self):
        """Save parameters and close the window (for OK button)."""
        self._ok_clicked = True
        success = self._save_parameters()
        if not success:
            self._ok_clicked = False  # Reset if save failed
        # Window will be closed in _save_parameters if save was successful
    
    def _auto_save(self):
        """Auto-save function called by timer."""
        if self.auto_save_enabled and self.root and self.root.winfo_exists():
            success = self._save_parameters()
            if success:
                logger.info("Auto-save completed successfully")
            # Schedule next auto-save
            self._auto_save_timer = self.root.after(self.save_delay, self._auto_save)
    
    def _create_gui(self):
        """Create the GUI interface."""
        # Create main window
        self.root = tk.Tk()
        self.root.title("Parameter Tuner")
        self.root.geometry("700x600")
        
        # Configure Times New Roman font for ttk styles
        style = ttk.Style()
        style.configure("Times.TLabelframe.Label", font=("Times New Roman", 10, "bold"))
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_rowconfigure(1, weight=1)  # Make scrollable area expandable
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Create save path display frame with Times New Roman font
        save_frame = ttk.LabelFrame(main_frame, text="Save Configuration", padding=10, style="Times.TLabelframe")
        save_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        save_frame.grid_columnconfigure(1, weight=1)
        
        # Save path label with Times New Roman bold
        save_label = tk.Label(save_frame, text="Save Path:", font=("Times New Roman", 10, "bold"))
        save_label.grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        # Save path display
        self.save_path_var = tk.StringVar(value=str(self.save_path))
        save_path_entry = ttk.Entry(save_frame, textvariable=self.save_path_var, state="readonly", width=50)
        save_path_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        
        # Browse button with Times New Roman bold
        browse_btn = tk.Button(save_frame, text="Browse", font=("Times New Roman", 9, "bold"), 
                              command=self._browse_save_path)
        browse_btn.grid(row=0, column=2, padx=(0, 5))
        
        # Reset to default button with Times New Roman bold
        reset_btn = tk.Button(save_frame, text="Reset to Default", font=("Times New Roman", 9, "bold"), 
                             command=self._reset_save_path)
        reset_btn.grid(row=0, column=3)
        
        # Create scrollable frame for parameters
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Grid layout for scrollable area
        canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")
        
        # Create parameter entries
        self.entries = {}
        for i, (key, action) in enumerate(self.args_defaults.items()):
            # Parameter name label with Times New Roman bold
            name_label = tk.Label(scrollable_frame, text=f"{key}:", font=("Times New Roman", 11, "bold"))
            name_label.grid(row=i*2, column=0, sticky="w", padx=(10, 5), pady=(10, 2))
            
            # Parameter description (if available) with Times New Roman
            if action.help:
                help_label = tk.Label(scrollable_frame, text=action.help, font=("Times New Roman", 9), 
                                     foreground="gray", wraplength=500)
                help_label.grid(row=i*2, column=1, columnspan=2, sticky="w", padx=(5, 10), pady=(10, 2))
            
            # Entry frame
            entry_frame = ttk.Frame(scrollable_frame)
            entry_frame.grid(row=i*2+1, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
            
            # Entry widget with activity tracking
            entry = ttk.Entry(entry_frame, width=60)
            entry.insert(0, str(action.default))
            entry.pack(side="left", fill="x", expand=True)
            
            # Bind entry events for activity tracking
            entry.bind('<Key>', lambda e: self._on_activity())
            entry.bind('<Button-1>', lambda e: self._on_activity())
            entry.bind('<FocusIn>', lambda e: self._on_activity())
            
            # Type indicator with Times New Roman
            type_label = tk.Label(entry_frame, text=f"({action.type.__name__ if action.type else 'str'})", 
                                 font=("Times New Roman", 8), foreground="blue")
            type_label.pack(side="right", padx=(5, 0))
            
            self.entries[key] = entry
        
        # Button frame
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=0, pady=(0, 10))
        
        # Save button with Times New Roman bold
        save_btn = tk.Button(button_frame, text="Save Parameters", font=("Times New Roman", 10, "bold"), 
                            command=self._save_parameters)
        save_btn.pack(side="left", padx=(0, 10))
        
        # OK button with Times New Roman bold (save and close)
        ok_btn = tk.Button(button_frame, text="OK", font=("Times New Roman", 10, "bold"), 
                          command=self._save_and_close)
        ok_btn.pack(side="left", padx=(0, 10))
        
        # Auto-save toggle with Times New Roman
        self.auto_save_var = tk.BooleanVar(value=self.auto_save_enabled)
        auto_save_check = tk.Checkbutton(button_frame, text="Auto-save", font=("Times New Roman", 9),
                                        variable=self.auto_save_var, command=self._toggle_auto_save)
        auto_save_check.pack(side="left", padx=(0, 10))
        
        # Status label with Times New Roman
        self.status_label = tk.Label(button_frame, text="Ready", font=("Times New Roman", 8))
        self.status_label.pack(side="right")
        
        # Bind window events for activity tracking
        self.root.bind('<Key>', lambda e: self._on_activity())
        self.root.bind('<Button-1>', lambda e: self._on_activity())
        self.root.bind('<Motion>', lambda e: self._on_activity())
        
        # Bind button events for activity tracking
        browse_btn.bind('<Button-1>', lambda e: self._on_activity())
        reset_btn.bind('<Button-1>', lambda e: self._on_activity())
        save_btn.bind('<Button-1>', lambda e: self._on_activity())
        ok_btn.bind('<Button-1>', lambda e: self._on_activity())
        auto_save_check.bind('<Button-1>', lambda e: self._on_activity())
        
        # Start timers
        if self.auto_save_enabled:
            self._auto_save_timer = self.root.after(self.save_delay, self._auto_save)
        
        # Start inactivity timer
        self._reset_inactivity_timer()
    
    def _browse_save_path(self):
        """Open file dialog to select save path."""
        new_path = filedialog.askdirectory(
            initialdir=str(self.save_path),
            title="Select Save Directory"
        )
        if new_path:
            self.save_path = Path(new_path)
            self.save_path.mkdir(parents=True, exist_ok=True)
            self.save_path_var.set(str(self.save_path))
            self.status_label.config(text=f"Save path updated: {self.save_path}", font=("Times New Roman", 8))
    
    def _reset_save_path(self):
        """Reset save path to default location."""
        default_path = Path.cwd()
        self.save_path = default_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_path_var.set(str(self.save_path))
        self.status_label.config(text=f"Reset to default: {self.save_path}", font=("Times New Roman", 8))
        self._show_message("Reset Complete", f"Save path reset to default:\n{self.save_path}")
    
    def _on_activity(self):
        """Record user activity and reset inactivity timer."""
        self._last_activity = time.time()
        self._reset_inactivity_timer()
    
    def _reset_inactivity_timer(self):
        """Reset the inactivity timer."""
        if self._inactivity_timer:
            self.root.after_cancel(self._inactivity_timer)
        self._inactivity_timer = self.root.after(self.inactivity_timeout * 1000, self._on_inactivity_timeout)
    
    def _on_inactivity_timeout(self):
        """Handle inactivity timeout - auto-save and close."""
        if self.root and self.root.winfo_exists():
            logger.info(f"No activity for {self.inactivity_timeout} seconds, auto-saving and closing")
            self.status_label.config(text="Auto-saving due to inactivity...", font=("Times New Roman", 8))
            self.root.update_idletasks()
            
            # Auto-save current parameters
            success = self._save_parameters()
            if success:
                self.status_label.config(text="Auto-saved and closing...", font=("Times New Roman", 8))
                self.root.update_idletasks()
                time.sleep(0.5)  # Brief pause to show status
            
            # Close the window
            self.root.quit()
    
    def _toggle_auto_save(self):
        """Toggle auto-save functionality."""
        self.auto_save_enabled = self.auto_save_var.get()
        if self.auto_save_enabled and self._auto_save_timer is None:
            self._auto_save_timer = self.root.after(self.save_delay, self._auto_save)
            self.status_label.config(text="Auto-save enabled", font=("Times New Roman", 8))
        elif not self.auto_save_enabled and self._auto_save_timer:
            self.root.after_cancel(self._auto_save_timer)
            self._auto_save_timer = None
            self.status_label.config(text="Auto-save disabled", font=("Times New Roman", 8))
    
    def tune(self) -> argparse.ArgumentParser:
        """
        Launch the parameter tuning GUI.
        
        Returns:
            Updated argparse parser with modified default values
        """
        try:
            self._create_gui()
            self.root.mainloop()
        except Exception as e:
            logger.error(f"GUI error: {e}")
            raise
        finally:
            # Clean up timers
            if self._auto_save_timer:
                self.root.after_cancel(self._auto_save_timer)
            if self._inactivity_timer:
                self.root.after_cancel(self._inactivity_timer)
        
        return self.parser
    
    @classmethod
    def tune_parameters(
        cls,
        parser: argparse.ArgumentParser,
        save_delay: int = 30000,
        save_path: Optional[str] = None,
        auto_save_enabled: bool = True,
        validation_callbacks: Optional[Dict[str, Callable]] = None,
        inactivity_timeout: int = 10
    ) -> argparse.ArgumentParser:
        """
        Convenience method to create and launch a ParameterTuner.
        
        Args:
            parser: ArgumentParser object to tune
            save_delay: Auto-save delay in milliseconds (default: 30 seconds)
            save_path: Directory to save parameters (default: current working directory)
            auto_save_enabled: Whether to enable auto-save
            validation_callbacks: Custom validation functions for parameters
            inactivity_timeout: Seconds of inactivity before auto-save and close
            
        Returns:
            Updated argparse parser with modified default values
        """
        tuner = cls(
            parser=parser,
            save_delay=save_delay,
            save_path=save_path,
            auto_save_enabled=auto_save_enabled,
            validation_callbacks=validation_callbacks,
            inactivity_timeout=inactivity_timeout
        )
        return tuner.tune()


# Backward compatibility
ParameterAdjuster = ParameterTuner

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example parameter adjustment.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    # Adjust parameters and get the updated parser
    updated_parser = ParameterAdjuster.adjust_parameters_with_gui(parser, save_path="output_params")
    print("Final Parser Defaults:")
    for action in updated_parser._actions:
        if action.dest != 'help':
            print(f"{action.dest}: {action.default}")
    parser = updated_parser.parse_args()

    print("Final Parser:")
    print(parser)