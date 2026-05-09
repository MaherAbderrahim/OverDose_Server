import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from scan.models import Scan

from .models import RiskAssessment, RiskItem
from .serializers import RiskRequestSerializer, RiskResponseSerializer


logger = logging.getLogger(__name__)


def build_mock_risks(scan_id, ingredients):
	levels = ["low", "medium", "high"]
	risks = [
		{
			"ingredient": ingredient,
			"level": levels[index % len(levels)],
		}
		for index, ingredient in enumerate(ingredients)
	]
	return {"scan_id": scan_id, "risks": risks}


def _persist_risks(scan_id, risks):
	scan = Scan.objects.filter(id=scan_id).first()
	if not scan:
		return

	assessment, _ = RiskAssessment.objects.get_or_create(scan=scan)
	assessment.payload = {"scan_id": scan_id, "risks": risks}
	assessment.save(update_fields=["payload", "updated_at"])
	assessment.items.all().delete()
	RiskItem.objects.bulk_create(
		[
			RiskItem(
				assessment=assessment,
				ingredient=item["ingredient"],
				level=item["level"],
			)
			for item in risks
		]
	)


class RiskEvaluationAPIView(APIView):
	def post(self, request):
		serializer = RiskRequestSerializer(data=request.data)
		serializer.is_valid(raise_exception=True)

		data = serializer.validated_data
		payload = build_mock_risks(data["scan_id"], data["ingredients"])
		_persist_risks(data["scan_id"], payload["risks"])

		response_serializer = RiskResponseSerializer(data=payload)
		response_serializer.is_valid(raise_exception=True)
		return Response(response_serializer.data, status=status.HTTP_200_OK)


class RiskPredictAPIView(APIView):
	def post(self, request):
		serializer = RiskRequestSerializer(data=request.data)
		serializer.is_valid(raise_exception=True)

		data = serializer.validated_data
		try:
			from risk.services import analyze_ingredients_risks, get_global_agent

			agent = get_global_agent()
			risk_items, _, _, _ = analyze_ingredients_risks(
				data["ingredients"],
				user_type=request.user.user_type if request.user.is_authenticated else None,
				user_id=request.user.id if request.user.is_authenticated else None,
				product_id=data["scan_id"],
				agent=agent,
			)
		except Exception:
			logger.exception("Risk prediction failed")
			return Response(
				{"detail": "Risk prediction failed."},
				status=status.HTTP_500_INTERNAL_SERVER_ERROR,
			)

		_persist_risks(data["scan_id"], risk_items)

		response_payload = {"scan_id": data["scan_id"], "risks": risk_items}
		response_serializer = RiskResponseSerializer(data=response_payload)
		response_serializer.is_valid(raise_exception=True)
		return Response(response_serializer.data, status=status.HTTP_200_OK)
