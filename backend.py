"""
Backend Module — Computational Logic
=====================================
Handles all processing logic for the app.
The frontend sends the user's name here, and this module
processes it and returns the formatted output.
"""


def process_username(name: str) -> dict:
    """
    Core computation: takes a username, validates and processes it,
    then returns structured output for the frontend to display.

    Args:
        name: The raw name string from user input.

    Returns:
        A dict with the greeting message and metadata.
    """
    # --- Input validation ---
    cleaned_name = name.strip()

    if not cleaned_name:
        return {
            "success": False,
            "message": "",
            "display_name": "",
            "error": "Name cannot be empty. Please enter a valid name.",
        }

    # --- Processing logic ---
    display_name = cleaned_name.title()  # Capitalize properly
    greeting_message = f"Hello, {display_name}! 👋 Welcome to the Dashboard."

    return {
        "success": True,
        "message": greeting_message,
        "display_name": display_name,
        "error": None,
    }
