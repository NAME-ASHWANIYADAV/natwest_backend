# Quiz Application Backend

This is the backend server for a quiz application, built with Python using the FastAPI framework. It uses Google Firestore for data storage.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
* Python 3.10+
* Git

---

## 🚀 Setup and Installation

Follow these steps to get your development environment set up.

### 1. Clone the Repository
First, clone the repository to your local machine.
```bash
git clone [https://github.com/NAME-ASHWANIYADAV/natwest_backend.git](https://github.com/NAME-ASHWANIYADAV/natwest_backend.git)
cd natwest_backend
```

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

* **Create the environment:**
    ```bash
    python -m venv venv
    ```
* **Activate the environment:**
    * On Windows:
        ```powershell
        .\venv\Scripts\Activate
        ```
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

### 3. Install Dependencies
Install all the required Python packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Configure Environment & Credentials (Crucial Step)

This application requires credentials to connect to Google Firestore.

* **Google Cloud Service Account:**
    1.  Go to your [Google Cloud Platform (GCP) Console](https://console.cloud.google.com/).
    2.  Navigate to **IAM & Admin** > **Service Accounts** and create a service account for this project.
    3.  Grant the service account the **"Cloud Datastore User"** role (or a more specific Firestore role).
    4.  Create a JSON key for this service account, download it, and save it in the root of this project folder.
    5.  **Rename the downloaded file to `serviceAccountKey.json`**.

* **Environment Variables File (`.env`):**
    1.  Create a file named `.env` in the root of the project directory.
    2.  Add any necessary environment variables to this file (e.g., database names, secret keys). If there are no other variables, this file can be left empty for now, but it's good practice to create it.

Your project directory should now look like this:
```
/natwest_backend
|-- .git/
|-- venv/
|-- __pycache__/
|-- main.py
|-- requirements.txt
|-- .gitignore
|-- serviceAccountKey.json  <-- Your GCP credentials
|-- .env                    <-- Your environment variables
```

---

## ▶️ Running the Application

Once the setup is complete, you can run the backend server using Uvicorn. The `--reload` flag will automatically restart the server whenever you make changes to the code.

```bash
uvicorn main:app --reload
```

The server will start, and you should see output in your terminal indicating that the application is running, typically on `http://127.0.0.1:8000`.
