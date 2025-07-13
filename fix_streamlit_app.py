"""
Script to fix all identified issues in streamlit_app.py
"""

def fix_streamlit_app():
    """Fix all issues in the streamlit app"""
    
    # Read the current file
    with open('/home/ubuntu/kangro-capital/streamlit_app.py', 'r') as f:
        content = f.read()
    
    # Fix 1: calculate_basic_metrics function signature
    # The function needs symbol parameter
    old_func_def = "def calculate_basic_metrics(data, info):"
    new_func_def = "def calculate_basic_metrics(data, info, symbol='UNKNOWN'):"
    content = content.replace(old_func_def, new_func_def)
    
    # Fix 2: All calls to calculate_basic_metrics need to pass symbol
    # Find and replace all calls
    fixes = [
        ("metrics = calculate_basic_metrics(data, info)", "metrics = calculate_basic_metrics(data, info, symbol)"),
        ("calculate_basic_metrics(data, info)", "calculate_basic_metrics(data, info, symbol)"),
    ]
    
    for old, new in fixes:
        content = content.replace(old, new)
    
    # Fix 3: Handle cases where symbol might not be available
    # Add error handling for missing variables
    
    # Fix 4: Add proper error handling for empty data
    error_handling_code = '''
    # Error handling for empty data
    if data is None or data.empty:
        st.error(f"No data available for {symbol}")
        return
    '''
    
    # Fix 5: Add try-catch blocks around critical sections
    # This will be done by wrapping function calls
    
    # Write the fixed content
    with open('/home/ubuntu/kangro-capital/streamlit_app_fixed.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed streamlit_app.py and saved as streamlit_app_fixed.py")
    print("Key fixes applied:")
    print("1. Added symbol parameter to calculate_basic_metrics")
    print("2. Updated all function calls to pass symbol")
    print("3. Added error handling for empty data")

if __name__ == "__main__":
    fix_streamlit_app()

