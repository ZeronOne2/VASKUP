"""
VASKUP Main Application Entry Point

This file serves as the main entry point for the VASKUP patent analysis system.
"""

import sys
from pathlib import Path

# Add src to Python path for development
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Load environment variables safely
try:
    from dotenv import load_dotenv

    load_dotenv(encoding="utf-8")
    print("✓ Environment variables loaded from .env file")
except Exception as e:
    print(f"⚠️ Warning: Could not load .env file: {e}")
    print("Continuing with system environment variables...")

try:
    from vaskup import __version__, __description__
    from vaskup.core import VASKUPCore
    from vaskup.utils import (
        setup_logging,
        validate_api_keys,
        ensure_directories,
        print_model_config,
    )

    print(f"VASKUP {__version__}")
    print(f"{__description__}")
    print()

    # Setup logging
    setup_logging("INFO")

    # Ensure directories exist
    ensure_directories()

    # Initialize core system
    core = VASKUPCore()
    health = core.health_check()

    print("System Health Check:")
    for key, value in health.items():
        print(f"  {key}: {value}")

    # Print model configuration
    print()
    print_model_config()

    # Validate API keys
    api_status = validate_api_keys()
    print("\nAPI Key Validation:")
    for service, is_valid in api_status.items():
        status = "✓ Available" if is_valid else "✗ Missing"
        print(f"  {service.upper()}: {status}")

    if all(api_status.values()):
        print("\n🎉 All API keys are configured!")
    else:
        print("\n⚠️  Some API keys are missing. Please check your .env file.")

    print("\n✅ VASKUP initialization completed successfully!")
    print("Ready for patent analysis workflows.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed with: uv sync")
    sys.exit(1)
except Exception as e:
    print(f"❌ Initialization error: {e}")
    sys.exit(1)
