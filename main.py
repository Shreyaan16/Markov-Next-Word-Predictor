import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import json

# Import our Markov model class and visualization functions
from predictor import (
    MarkovChainTextPredictor,
    LARGE_CORPUS,
    visualize_transition_matrix,
    visualize_stationary_distribution,
    visualize_convergence,
    visualize_chain_properties
)

# --- App Setup ---
app = FastAPI(
    title="Applied Stochastic Models: Markov Chain Dashboard",
    description="A project demonstrating Markov chain concepts for next-word prediction."
)

# Mount a 'static' directory to serve images, CSS, etc.
STATIC_DIR = "static"
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup HTML templates
templates = Jinja2Templates(directory="templates")

# --- Global Model ---
# We'll use an Order-1/2 model for a better demo.
model = MarkovChainTextPredictor(order=2)

@app.on_event("startup")
def startup_event():
    """
    On server startup:
    1. Train the Markov model.
    2. Pre-generate all visualization graphs and save them as static files.
    """
    print("--- Server Startup ---")
    print(f"Training Markov model (Order {model.order})...")
    model.build_model(LARGE_CORPUS)
    print("Model training complete.")
    
    print("Generating static analysis graphs...")
    try:
        visualize_transition_matrix(model, top_n=20, save_path=os.path.join(IMAGES_DIR, "transition_matrix.png"))
        visualize_stationary_distribution(model, top_n=20, save_path=os.path.join(IMAGES_DIR, "stationary_dist.png"))
        visualize_convergence(model, steps=300, save_path=os.path.join(IMAGES_DIR, "convergence.png"))
        visualize_chain_properties(model, save_path=os.path.join(IMAGES_DIR, "chain_properties.png"))
        print("All graphs generated successfully.")
    except Exception as e:
        print(f"Error generating graphs: {e}")
    print("--- Server Ready ---")

# --- Pydantic Models for API ---
class PredictRequest(BaseModel):
    context: str

class GenerateRequest(BaseModel):
    length: int = 50

# === API Endpoints ===

@app.get("/", response_class=HTMLResponse)
async def get_predictor_page(request: Request):
    """Serves the main next-word predictor page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Next-Word Predictor",
        "model_order": model.order
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard_page(request: Request):
    """
    Serves the main analysis dashboard for your professor.
    It runs all the mathematical analyses live and passes
    the results to the HTML template.
    """
    
    # --- Run all analyses from the slides ---
    
    # --- THIS IS THE FIX ---
    # Define sample states for an Order-1 model
    start_state = 'computer'
    end_state = 'is'
    
    # 1. n-Step Probability (P^n)
    n_step_res = model.get_n_step_probability(start_state, end_state, n=2)
    
    # 2. Chapman-Kolmogorov
    ck_res = model.demonstrate_chapman_kolmogorov(n=2, m=3, start_state=start_state, end_state=end_state)
    
    # 3. Class Structure
    class_res = model.get_communicating_classes()
    
    # 4. Ergodicity (Period + Recurrence)
    period_res = model.get_period(start_state)
    recur_res = model.compute_expected_return_time(start_state)
    
    # 5. Stationary Distribution (Data)
    stationary_res = model.compute_stationary_distribution(top_n=10)

    # --- THIS IS THE FIX (Part 2) ---
    # Define string variables *before* the dictionary
    start_state_str = str(start_state)
    end_state_str = str(end_state)

    # We use json.dumps for pretty-printing the dicts in HTML
    context_data = {
        "request": request,
        "title": "Analysis Dashboard",
        "model_order": model.order,
        "start_state_str": start_state_str,  # Now we just pass the variable
        "end_state_str": end_state_str,      # Now we just pass the variable
        # This line will now work correctly
        "n_step_res": f"Pr(X_2 = '{end_state_str}' | X_0 = '{start_state_str}') = {n_step_res:.6f}",
        "ck_res": json.dumps(ck_res, indent=2),
        "class_res": json.dumps(class_res, indent=2),
        "period_res": period_res,
        "recur_res": f"{recur_res:.2f} steps",
        "stationary_res": stationary_res.to_json(indent=2)
    }
    
    return templates.TemplateResponse("dashboard.html", context_data)


@app.post("/api/predict")
async def api_predict(data: PredictRequest):
    """API endpoint to get next word predictions."""
    try:
        predictions = model.predict_next_word(data.context, top_k=5)
        return {"success": True, "predictions": predictions}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/generate")
async def api_generate(data: GenerateRequest):
    """API endpoint to generate a block of text."""
    try:
        text = model.generate_text(length=data.length)
        return {"success": True, "text": text}
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- Run the App ---
if __name__ == "__main__":
    print("Starting Uvicorn server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)