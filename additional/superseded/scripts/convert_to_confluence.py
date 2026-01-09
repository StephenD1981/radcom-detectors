#!/usr/bin/env python3
"""
Convert Obsidian markdown files to Confluence-compatible format.

Usage:
    python scripts/convert_to_confluence.py obsidian/production/ confluence_output/

Handles:
- [[wiki links]] → [page title] or inline text
- ```code blocks``` → {code} macros
- Tables (already compatible)
- Mermaid diagrams → {code:mermaid} or placeholder
- Canvas files → Skipped (not portable)
"""
import re
import sys
from pathlib import Path
import shutil


def convert_wiki_links(content: str) -> str:
    """Convert [[Page Name]] to [Page Name] for Confluence."""
    # [[PAGE]] → [PAGE]
    content = re.sub(r'\[\[([^\]|]+)\]\]', r'[\1]', content)
    # [[PAGE|alias]] → [alias|PAGE]
    content = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'[\2|\1]', content)
    return content


def convert_code_blocks(content: str) -> str:
    """Convert markdown code blocks to Confluence {code} macro."""
    # ```language\ncode\n``` → {code:language}\ncode\n{code}
    def replace_code(match):
        lang = match.group(1) or 'none'
        code = match.group(2)
        # Map common languages
        lang_map = {
            'bash': 'bash',
            'python': 'python',
            'json': 'javascript',
            'yaml': 'yaml',
            'sql': 'sql',
            'javascript': 'javascript',
            'js': 'javascript',
            'ini': 'none',
            'nginx': 'none',
            '': 'none',
        }
        confluence_lang = lang_map.get(lang.lower(), lang)
        return f'{{code:{confluence_lang}}}\n{code}{{code}}'

    content = re.sub(
        r'```(\w*)\n(.*?)```',
        replace_code,
        content,
        flags=re.DOTALL
    )
    return content


def convert_callouts(content: str) -> str:
    """Convert Obsidian callouts to Confluence panels."""
    # > [!NOTE] → {info}
    # > [!WARNING] → {warning}
    # > [!TIP] → {tip}
    callout_map = {
        'NOTE': 'info',
        'INFO': 'info',
        'TIP': 'tip',
        'WARNING': 'warning',
        'CAUTION': 'warning',
        'IMPORTANT': 'note',
        'DANGER': 'warning',
    }

    for obsidian_type, confluence_type in callout_map.items():
        pattern = rf'> \[!{obsidian_type}\](.*?)\n((?:>.*\n)*)'
        def replace_callout(match):
            title = match.group(1).strip()
            body = match.group(2)
            # Remove > prefix from each line
            body = re.sub(r'^> ?', '', body, flags=re.MULTILINE)
            if title:
                return f'{{{confluence_type}:title={title}}}\n{body}{{{confluence_type}}}\n'
            return f'{{{confluence_type}}}\n{body}{{{confluence_type}}}\n'
        content = re.sub(pattern, replace_callout, content, flags=re.IGNORECASE)

    return content


def convert_task_lists(content: str) -> str:
    """Convert markdown task lists."""
    # - [ ] task → * ( ) task
    # - [x] task → * (/) task
    content = re.sub(r'^- \[ \] ', '* ( ) ', content, flags=re.MULTILINE)
    content = re.sub(r'^- \[x\] ', '* (/) ', content, flags=re.MULTILINE)
    return content


def convert_highlights(content: str) -> str:
    """Convert ==highlights== to bold (Confluence doesn't have highlight)."""
    content = re.sub(r'==([^=]+)==', r'*\1*', content)
    return content


def convert_file(input_path: Path, output_path: Path) -> bool:
    """Convert a single file."""
    if input_path.suffix == '.canvas':
        print(f"  Skipping canvas file: {input_path.name}")
        return False

    if input_path.suffix != '.md':
        print(f"  Skipping non-markdown: {input_path.name}")
        return False

    content = input_path.read_text(encoding='utf-8')

    # Apply conversions
    content = convert_wiki_links(content)
    content = convert_code_blocks(content)
    content = convert_callouts(content)
    content = convert_task_lists(content)
    content = convert_highlights(content)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding='utf-8')

    return True


def convert_directory(input_dir: Path, output_dir: Path):
    """Convert all files in a directory."""
    print(f"Converting {input_dir} → {output_dir}")

    converted = 0
    skipped = 0

    for input_path in input_dir.glob('*'):
        if input_path.is_file():
            output_path = output_dir / input_path.name
            if convert_file(input_path, output_path):
                print(f"  ✓ {input_path.name}")
                converted += 1
            else:
                skipped += 1

    print(f"\nDone: {converted} converted, {skipped} skipped")
    print(f"\nOutput in: {output_dir}")
    print("\nTo import to Confluence:")
    print("  1. Create a new page in Confluence")
    print("  2. Click ⋮ → Import → Markdown")
    print("  3. Paste the converted content")
    print("  Or use the Confluence REST API for bulk import")


def main():
    if len(sys.argv) < 2:
        # Default paths
        input_dir = Path("obsidian/production")
        output_dir = Path("docs/confluence")
    elif len(sys.argv) == 2:
        input_dir = Path(sys.argv[1])
        output_dir = Path("docs/confluence")
    else:
        input_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    convert_directory(input_dir, output_dir)


if __name__ == '__main__':
    main()
