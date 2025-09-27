from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from app.pokemon_card_identifier_poketracker_COMBINED_fixed import VisualCardIdentifier  # Assuming your identifier class is saved here

class CardIdentifyView(APIView):
    parser_classes = [MultiPartParser]

    print(parser_classes)

    def post(self, request, *args, **kwargs):

        card_image = request.FILES.get('image')  # Expecting an image file

        if not card_image:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Save the image to a temporary location for processing
        image_path = f"temp_images/{card_image.name}"
        with open(image_path, 'wb') as f:
            for chunk in card_image.chunks():
                f.write(chunk)
        
        try:
            # Initialize the identifier system
            identifier = VisualCardIdentifier(model_type="clip")  # or "dinov2"
            matches = identifier.identify_card(image_path)

            if matches:
                response_data = [{"name": match.name, "confidence": match.confidence, "price": match.pricing.market_price if match.pricing else "N/A"} for match in matches]
                return Response(response_data, status=status.HTTP_200_OK)
            else:
                return Response({"message": "No matches found"}, status=status.HTTP_404_NOT_FOUND)
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
