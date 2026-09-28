"""
Backend Module — Core Logic
============================
Handles name processing and greeting generation.
The frontend calls this module and displays the returned message.
"""


def generate_greeting(user_name: str) -> str:
    """
    Process the user-provided name and build a personalised greeting.

    Args:
        user_name: Raw name string coming from the frontend input.

    Returns:
        A greeting string ready for display.
    """
    sanitised = (user_name or "").strip()

    if not sanitised:
        return "Hey there! Drop your name in the box above so we can say a proper hello."

    # Capitalise each word for a polished look
    formatted = sanitised.title()
    return f"Hello, {formatted}! Great to have you on the dashboard. 🎉"
