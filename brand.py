"""
brand.py — SINGLE source of truth for the app's name + colors.
Edit ONLY this file to re-skin the whole app. Every value is (light, dark)
unless noted. customtkinter picks the right one from the appearance mode.
"""

# ---- app identity --------------------------------------------------------
APP_NAME = "Mixture Classifier"
APP_TAGLINE = "pure spectra → detect + ratio"

# ---- surfaces (aligned to the UNMIXR light family; see family.py) --------
SIDE_FILL = ("#eef1f5", "#12161c")     # sidebar background
MAIN_FILL = ("#f5f7fa", "#0f1216")     # dashboard background
CARD_FILL = ("#ffffff", "#171c22")     # card background

# ---- text ----------------------------------------------------------------
TEXT     = ("#1c2430", "#e6edf3")      # primary text (usually leave to theme)
SUBTLE   = ("#5b6673", "#9ca3af")      # secondary/hint text

# ---- buttons / accents ---------------------------------------------------
BTN_FILL   = ("#1a73e8", "#2f6fe0")    # normal button (family blue)
BTN_HOVER  = ("#155ec2", "#255bbd")    # button hover
RUN_FILL   = ("#0f9d6b", "#12b866")    # the primary "Run" button (family teal)
RUN_HOVER  = ("#0c855a", "#0e9a55")
ACCENT     = "#0f9d6b"                  # misc accent (family teal)

GOOD = "#1a8f5f"
WARN = "#c2551f"

# ---- matplotlib series palette (per-component), colorblind-safe ----------
SERIES = ["#4c78a8", "#f58518", "#54a24b", "#e45756",
          "#72b7b2", "#b279a2", "#ff9da6", "#9d755d"]
