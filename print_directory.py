#!/usr/bin/env python3
"""
Directory Tree Explorer - A powerful terminal-based directory structure visualizer
Features:
- Multiple tree styles (ASCII, Unicode, Compact)
- File size display with human-readable formatting
- File type filtering and exclusion
- Color coding by file type
- Statistics summary
- Depth limiting
- Hidden file handling
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import stat


class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # File type colors
    DIR = '\033[94m'        # Blue
    EXEC = '\033[92m'       # Green
    IMAGE = '\033[95m'      # Magenta
    ARCHIVE = '\033[93m'    # Yellow
    CODE = '\033[96m'       # Cyan
    TEXT = '\033[97m'       # White
    DEFAULT = '\033[0m'     # Default


class TreeStyle:
    """Different tree visualization styles"""
    ASCII = {
        'branch': '├── ',
        'last': '└── ',
        'pipe': '│   ',
        'space': '    '
    }
    
    UNICODE = {
        'branch': '├─ ',
        'last': '└─ ',
        'pipe': '│  ',
        'space': '   '
    }
    
    COMPACT = {
        'branch': '├',
        'last': '└',
        'pipe': '│',
        'space': ' '
    }


class DirectoryExplorer:
    def __init__(self, style='unicode', show_size=True, show_hidden=False, 
                 max_depth=None, use_colors=True, filters=None, exclude=None):
        self.style = getattr(TreeStyle, style.upper(), TreeStyle.UNICODE)
        self.show_size = show_size
        self.show_hidden = show_hidden
        self.max_depth = max_depth
        self.use_colors = use_colors and sys.stdout.isatty()
        self.filters = filters or []
        self.exclude = exclude or []
        
        # Statistics
        self.stats = {
            'dirs': 0,
            'files': 0,
            'total_size': 0,
            'largest_file': ('', 0),
            'file_types': {}
        }
        
        # File type extensions
        self.file_types = {
            'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico', '.webp'},
            'archive': {'.zip', '.rar', '.tar', '.gz', '.7z', '.bz2', '.xz'},
            'code': {'.py', '.js', '.html', '.css', '.cpp', '.c', '.java', '.go', '.rs', '.php'},
            'text': {'.txt', '.md', '.rst', '.log', '.json', '.xml', '.yaml', '.yml', '.csv'}
        }

    def get_file_color(self, path):
        """Get color code for file based on type"""
        if not self.use_colors:
            return Colors.DEFAULT
            
        if path.is_dir():
            return Colors.DIR
        elif path.stat().st_mode & stat.S_IEXEC:
            return Colors.EXEC
        
        ext = path.suffix.lower()
        for file_type, extensions in self.file_types.items():
            if ext in extensions:
                return getattr(Colors, file_type.upper(), Colors.DEFAULT)
        
        return Colors.DEFAULT

    def format_size(self, size):
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:3.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}PB"

    def should_show_item(self, path):
        """Determine if item should be displayed based on filters"""
        name = path.name
        
        # Check hidden files
        if not self.show_hidden and name.startswith('.'):
            return False
        
        # Check exclusion patterns
        for pattern in self.exclude:
            if pattern in name or name.endswith(pattern):
                return False
        
        # Check inclusion filters (if any specified)
        if self.filters:
            for pattern in self.filters:
                if pattern in name or name.endswith(pattern):
                    return True
            return False
        
        return True

    def update_stats(self, path):
        """Update statistics for the current path"""
        if path.is_dir():
            self.stats['dirs'] += 1
        else:
            self.stats['files'] += 1
            try:
                size = path.stat().st_size
                self.stats['total_size'] += size
                
                if size > self.stats['largest_file'][1]:
                    self.stats['largest_file'] = (str(path), size)
                
                # Track file types
                ext = path.suffix.lower()
                if ext:
                    self.stats['file_types'][ext] = self.stats['file_types'].get(ext, 0) + 1
            except (OSError, IOError):
                pass

    def print_tree(self, directory, prefix="", is_last=True, current_depth=0):
        """Recursively print directory tree"""
        path = Path(directory)
        
        if not path.exists():
            print(f"Error: Directory '{directory}' does not exist.")
            return
        
        if not path.is_dir():
            print(f"Error: '{directory}' is not a directory.")
            return
        
        # Print current directory
        if current_depth == 0:
            color = self.get_file_color(path)
            size_info = ""
            if self.show_size:
                try:
                    total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    size_info = f" [{self.format_size(total_size)}]"
                except (OSError, IOError):
                    size_info = " [?]"
            
            print(f"{color}{Colors.BOLD}{path.name}/{Colors.RESET}{size_info}")
            self.update_stats(path)
        
        # Check depth limit
        if self.max_depth is not None and current_depth >= self.max_depth:
            return
        
        try:
            # Get and sort directory contents
            items = [p for p in path.iterdir() if self.should_show_item(p)]
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
            
            for i, item in enumerate(items):
                is_last_item = i == len(items) - 1
                
                # Determine prefix for current item
                if is_last_item:
                    current_prefix = prefix + self.style['last']
                    next_prefix = prefix + self.style['space']
                else:
                    current_prefix = prefix + self.style['branch']
                    next_prefix = prefix + self.style['pipe']
                
                # Get item info
                color = self.get_file_color(item)
                name = item.name
                suffix = "/" if item.is_dir() else ""
                
                size_info = ""
                if self.show_size and item.is_file():
                    try:
                        size = item.stat().st_size
                        size_info = f" [{self.format_size(size)}]"
                    except (OSError, IOError):
                        size_info = " [?]"
                
                # Print item
                print(f"{current_prefix}{color}{name}{suffix}{Colors.RESET}{size_info}")
                
                # Update statistics
                self.update_stats(item)
                
                # Recurse into directories
                if item.is_dir():
                    self.print_tree(item, next_prefix, is_last_item, current_depth + 1)
                    
        except PermissionError:
            print(f"{prefix}[Permission Denied]")
        except OSError as e:
            print(f"{prefix}[Error: {e}]")

    def print_statistics(self):
        """Print directory statistics summary"""
        if not self.use_colors:
            return
            
        print(f"\n{Colors.BOLD}📊 Directory Statistics:{Colors.RESET}")
        print(f"  Directories: {Colors.DIR}{self.stats['dirs']}{Colors.RESET}")
        print(f"  Files: {Colors.DEFAULT}{self.stats['files']}{Colors.RESET}")
        print(f"  Total Size: {Colors.BOLD}{self.format_size(self.stats['total_size'])}{Colors.RESET}")
        
        if self.stats['largest_file'][0]:
            print(f"  Largest File: {self.stats['largest_file'][0]} "
                  f"[{self.format_size(self.stats['largest_file'][1])}]")
        
        # Show top file types
        if self.stats['file_types']:
            sorted_types = sorted(self.stats['file_types'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top Extensions: {', '.join([f'{ext}({count})' for ext, count in sorted_types])}")


def main():
    parser = argparse.ArgumentParser(
        description="Display directory tree structure with advanced options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Show current directory
  %(prog)s /path/to/dir             # Show specific directory
  %(prog)s -d 3 -s                  # Limit depth to 3 levels, hide sizes
  %(prog)s --style ascii --hidden   # Use ASCII style, show hidden files
  %(prog)s -f .py -f .js            # Only show Python and JavaScript files
  %(prog)s -x node_modules -x .git  # Exclude specific directories
        """)
    
    parser.add_argument('directory', nargs='?', default='.', 
                       help='Directory to explore (default: current directory)')
    parser.add_argument('-s', '--no-size', action='store_true', 
                       help='Hide file sizes')
    parser.add_argument('--hidden', action='store_true', 
                       help='Show hidden files and directories')
    parser.add_argument('-d', '--depth', type=int, 
                       help='Maximum depth to traverse')
    parser.add_argument('--style', choices=['ascii', 'unicode', 'compact'], 
                       default='unicode', help='Tree style (default: unicode)')
    parser.add_argument('--no-color', action='store_true', 
                       help='Disable colored output')
    parser.add_argument('-f', '--filter', action='append', dest='filters',
                       help='Only show files matching pattern (can use multiple times)')
    parser.add_argument('-x', '--exclude', action='append', 
                       help='Exclude files/dirs matching pattern (can use multiple times)')
    parser.add_argument('--stats', action='store_true', 
                       help='Show directory statistics')
    
    args = parser.parse_args()
    
    explorer = DirectoryExplorer(
        style=args.style,
        show_size=not args.no_size,
        show_hidden=args.hidden,
        max_depth=args.depth,
        use_colors=not args.no_color,
        filters=args.filters,
        exclude=args.exclude
    )
    
    print(f"{Colors.BOLD}🌳 Directory Tree Explorer{Colors.RESET}")
    print(f"📁 Exploring: {os.path.abspath(args.directory)}")
    print(f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    explorer.print_tree(args.directory)
    
    if args.stats:
        explorer.print_statistics()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.DIM}Operation cancelled by user.{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.BOLD}Error: {e}{Colors.RESET}")
        sys.exit(1)