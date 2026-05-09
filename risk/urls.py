from django.urls import path

from .views import RiskEvaluationAPIView, RiskPredictAPIView

urlpatterns = [
    path("", RiskEvaluationAPIView.as_view(), name="risk-evaluate"),
    path("predict/", RiskPredictAPIView.as_view(), name="risk-predict"),
]