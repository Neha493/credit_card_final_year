# Credit Card Fraud Detection Website

## Author's Identity
- **Author:** Neha Sanwal
- **Supervisor:** [Supervisor's Name]

## Project Overview
This project is a web-based Credit Card Fraud Detection system that uses machine learning to identify potentially fraudulent transactions. The system provides a user-friendly interface for both administrators and regular users to monitor and manage credit card transactions.

## Files in the Project
1. **Frontend Files:**
   - `templates/dashboard.html` - Main dashboard interface
   - `static/app.css` - Main stylesheet
   - `static/app.js` - Main JavaScript file
   - `static/preloader.css` - Preloader animation styles
   - `static/preloader.js` - Preloader animation logic

2. **Backend Files:**
   - `app.py` - Main Flask application
   - `models/` - Machine learning model files
   - `templates/` - HTML templates
   - `static/` - Static assets (CSS, JS, images)

## Hardware Requirements
- **Processor:** Intel Core i3 or equivalent
- **RAM:** Minimum 4GB (8GB recommended)
- **Storage:** 500MB free space
- **Display:** 1366x768 minimum resolution
- **Internet Connection:** Required for real-time fraud detection
- **Graphics Card:** Not required (standard graphics sufficient)

## Software Requirements
1. **Operating System:**
   - Windows 10/11
   - Linux (Ubuntu 20.04 or later)
   - macOS 10.15 or later

2. **Python Environment:**
   - Python 3.8 or later
   - pip (Python package manager)

3. **Required Python Packages:**
   ```
   Flask==2.0.1
   scikit-learn==1.0.2
   pandas==1.3.3
   numpy==1.21.2
   ```

4. **Web Technologies:**
   - HTML5
   - CSS3
   - JavaScript (ES6+)
   - Bootstrap 5.3.0
   - Tailwind CSS 2.2.19

5. **Development Tools:**
   - Git (for version control)
   - VS Code or any modern IDE
   - Modern web browser (Chrome, Firefox, Safari, Edge)

## Installation Instructions
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Features
- Real-time credit card fraud detection
- User authentication system
- Admin dashboard
- Interactive UI with modern design
- Newsletter subscription
- Contact and support system

## Security
- Secure user authentication
- Encrypted data transmission
- Protected admin routes
- Machine learning-based fraud detection

## Acknowledgments
- Font Awesome for icons
- Google Fonts for typography
- Bootstrap and Tailwind CSS for UI components
