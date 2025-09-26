class CardModel {
  final String id;
  final String name;
  final String imageUrl;
  final List<double> prices;

  CardModel({
    required this.id,
    required this.name,
    required this.imageUrl,
    required this.prices,
  });

  factory CardModel.fromJson(Map<String, dynamic> json) {
    return CardModel(
      id: json["card_id"],
      name: json["name"],
      imageUrl: json["image_url"] ?? "",
      prices: (json["prices"] as List).map((e) => e.toDouble()).toList(),
    );
  }
}
