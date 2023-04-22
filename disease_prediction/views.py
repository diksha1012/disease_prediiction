from collections import defaultdict
from django.shortcuts import redirect, render

from disease_prediction.ml_alogorithm import predict_disease
from .form import SymptomsForm

from .chatGPT import ask_chatgpt

def home(request):
  if request.method == "GET":
    return render(request, 'disease_prediction/home.html', context={
      'form' : SymptomsForm()
    })
    
  else:
    form = SymptomsForm(request.POST)
    input_for_analyze = ""
    result = None
    if form.is_valid():
      symptoms = form.cleaned_data.get('symptoms')
      for symp in symptoms[0:len(symptoms)-1]:
        input_for_analyze += symp + ","
      input_for_analyze += symptoms[-1]
    predicted_disease = predict_disease(input_for_analyze)
    final_result = defaultdict(dict)
    for i, j in predicted_disease.items():
      final_result[i] = {
        "disease_name" : i,
        "percentage" : j,
        "symptoms" : ask_chatgpt("symptoms of", i),
        "precautions" : ask_chatgpt("precautions for", i)
      }
    return render(request, 'disease_prediction/result.html', context={
      'result' : final_result.values(),
      'symptoms' : input_for_analyze,
    })