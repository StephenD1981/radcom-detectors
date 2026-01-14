"""
Output generators for RAN Optimizer.

This package contains modules for generating various output formats,
including PostgreSQL-ready tables for production integration.
"""

from ran_optimizer.outputs.pg_tables_generator import PGTablesGenerator

__all__ = ['PGTablesGenerator']
