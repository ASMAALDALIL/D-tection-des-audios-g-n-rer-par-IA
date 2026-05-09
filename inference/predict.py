
import torch
from preprocessing.audio_processor import load_waveform, normalize_waveform
from data.dataset import waveform_to_melspec


def predict(model: torch.nn.Module, path: str, device: torch.device) -> float:
   
    model.eval()
    wav  = load_waveform(path)               
    wav  = normalize_waveform(wav)           
    spec = waveform_to_melspec(wav)          
    spec = spec.unsqueeze(0).to(device)    
    with torch.no_grad():
        logit = model(spec)                  
        prob  = torch.sigmoid(logit).item()  

    return prob


def batch_predict(model, paths, device, threshold=0.5):
   
    results = []
    for path in paths:
        try:
            score = predict(model, path, device)
            results.append({
                "path":  path,
                "score": score,
                "label": "spoof" if score >= threshold else "bonafide",
            })
        except Exception as e:
            results.append({"path": path, "score": None, "label": "error", "error": str(e)})
    return results
