"""
CLI entry point for ran-optimize command.

This provides a user-friendly command-line interface for the RAN optimizer.
"""
from ran_optimizer.runner import main

# Re-export main for the console_scripts entry point
__all__ = ['main']

if __name__ == '__main__':
    main()
