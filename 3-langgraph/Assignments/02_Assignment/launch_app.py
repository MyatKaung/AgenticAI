#!/usr/bin/env python3
"""
Launcher script for AI Travel Agent & Expense Planner
Allows users to choose between different interface options.
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if required environment variables are set"""
    required_keys = ["GROQ_API_KEY", "TAVILY_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    return missing_keys

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        "streamlit": "streamlit",
        "gradio": "gradio",
        "langchain": "langchain",
        "langgraph": "langgraph"
    }
    
    missing_packages = []
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def launch_streamlit():
    """Launch Streamlit app"""
    print("🚀 Launching Streamlit AI Travel Agent...")
    print("📱 The app will open in your default web browser at http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run(["streamlit", "run", "streamlit_travel_app.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to launch Streamlit. Make sure it's installed: pip install streamlit")
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped.")

def launch_gradio():
    """Launch Gradio app"""
    print("🚀 Launching Gradio AI Travel Agent...")
    print("📱 The app will open in your default web browser at http://localhost:7860")
    print("⏹️ Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run(["python", "gradio_travel_app.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to launch Gradio. Make sure it's installed: pip install gradio")
    except KeyboardInterrupt:
        print("\n👋 Gradio app stopped.")

def launch_jupyter():
    """Launch Jupyter notebook"""
    print("🚀 Launching Jupyter Notebook...")
    print("📱 The notebook will open in your default web browser")
    print("⏹️ Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run(["jupyter", "notebook", "ai_travel_agent_demo.ipynb"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to launch Jupyter. Make sure it's installed: pip install jupyter")
    except KeyboardInterrupt:
        print("\n👋 Jupyter server stopped.")

def run_tests():
    """Run the test script"""
    print("🧪 Running AI Travel Agent tests...\n")
    
    try:
        subprocess.run(["python", "test_travel_agent.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Tests failed. Check the error messages above.")

def run_command_line():
    """Run the command line version"""
    print("💻 Running AI Travel Agent in command line mode...\n")
    
    try:
        subprocess.run(["python", "ai_travel_agent.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to run command line version.")

def display_menu():
    """Display the main menu"""
    print("\n" + "=" * 60)
    print("✈️  AI TRAVEL AGENT & EXPENSE PLANNER")
    print("=" * 60)
    print("Choose your preferred interface:")
    print()
    print("1. 🌐 Streamlit Web App (Recommended)")
    print("   - Modern web interface with interactive forms")
    print("   - Real-time updates and beautiful visualizations")
    print("   - Best for general users")
    print()
    print("2. 🎨 Gradio Web App")
    print("   - Simple and clean web interface")
    print("   - Easy to use with tabbed results")
    print("   - Great for quick testing")
    print()
    print("3. 📓 Jupyter Notebook")
    print("   - Interactive development environment")
    print("   - Step-by-step examples and documentation")
    print("   - Best for developers and learning")
    print()
    print("4. 🧪 Run Tests")
    print("   - Validate system functionality")
    print("   - Check API keys and dependencies")
    print("   - Troubleshoot issues")
    print()
    print("5. 💻 Command Line")
    print("   - Terminal-based interface")
    print("   - Minimal dependencies")
    print("   - Best for automation")
    print()
    print("6. ❌ Exit")
    print()
    print("=" * 60)

def main():
    """Main launcher function"""
    # Check environment
    missing_keys = check_environment()
    if missing_keys:
        print("⚠️  WARNING: Missing API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n💡 Please set up your .env file with the required API keys.")
        print("   You can copy .env.template to .env and fill in your keys.")
        print("\n🔄 The apps will still launch but won't function properly without API keys.\n")
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print("⚠️  WARNING: Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Please install missing packages:")
        print("   pip install -r requirements.txt\n")
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['streamlit', 'st', '1']:
            launch_streamlit()
            return
        elif arg in ['gradio', 'gr', '2']:
            launch_gradio()
            return
        elif arg in ['jupyter', 'notebook', 'nb', '3']:
            launch_jupyter()
            return
        elif arg in ['test', 'tests', '4']:
            run_tests()
            return
        elif arg in ['cli', 'command', 'cmd', '5']:
            run_command_line()
            return
        elif arg in ['help', '-h', '--help']:
            print("Usage: python launch_app.py [option]")
            print("Options:")
            print("  streamlit, st, 1    - Launch Streamlit web app")
            print("  gradio, gr, 2       - Launch Gradio web app")
            print("  jupyter, nb, 3      - Launch Jupyter notebook")
            print("  test, tests, 4      - Run tests")
            print("  cli, cmd, 5         - Run command line version")
            print("  help, -h, --help    - Show this help message")
            return
    
    # Interactive menu
    while True:
        display_menu()
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                launch_streamlit()
                break
            elif choice == '2':
                launch_gradio()
                break
            elif choice == '3':
                launch_jupyter()
                break
            elif choice == '4':
                run_tests()
                input("\nPress Enter to return to menu...")
            elif choice == '5':
                run_command_line()
                input("\nPress Enter to return to menu...")
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()