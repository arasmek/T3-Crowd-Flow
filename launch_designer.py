"""
Launch PySide6 Designer
Run this script to open Qt Designer
"""
import sys
import os
import subprocess
from pathlib import Path

def find_designer():
    """Find PySide6 Designer executable"""
    try:
        import PySide6
        pyside_path = Path(PySide6.__file__).parent
        
        # Try different possible names
        possible_names = ['designer.exe', 'designer', 'Designer.exe']
        
        for name in possible_names:
            designer_path = pyside_path / name
            if designer_path.exists():
                return designer_path
        
        # Also check in Qt/bin subdirectory
        qt_bin = pyside_path / 'Qt' / 'bin'
        if qt_bin.exists():
            for name in possible_names:
                designer_path = qt_bin / name
                if designer_path.exists():
                    return designer_path
                    
    except ImportError:
        print("PySide6 is not installed!")
        print("Install it with: pip install PySide6")
        return None
    
    return None

def main():
    print("Searching for PySide6 Designer...")
    designer_path = find_designer()
    
    if designer_path:
        print(f"Found Designer at: {designer_path}")
        print("Launching...")
        
        try:
            # Launch Designer
            subprocess.Popen([str(designer_path)])
            print("Designer launched successfully!")
        except Exception as e:
            print(f"Error launching Designer: {e}")
            print("\nTry running it directly:")
            print(f'  "{designer_path}"')
    else:
        print("\n❌ Designer not found!")
        print("\nTroubleshooting:")
        print("1. Make sure PySide6 is installed:")
        print("   pip install PySide6")
        print("\n2. Try finding it manually:")
        print("   python -c \"import PySide6; print(PySide6.__file__)\"")
        print("\n3. Look in that folder for designer.exe")
        
        # Show where we looked
        try:
            import PySide6
            print(f"\n4. PySide6 is installed at:")
            print(f"   {Path(PySide6.__file__).parent}")
        except:
            pass

if __name__ == "__main__":
    main()
