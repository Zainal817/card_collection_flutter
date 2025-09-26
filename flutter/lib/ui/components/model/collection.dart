class CollectionItem {
  final String cardId;
  final String name;
  final String imageUrl;

  CollectionItem({
    required this.cardId,
    required this.name,
    required this.imageUrl,
  });

  factory CollectionItem.fromJson(Map<String, dynamic> json) {
    return CollectionItem(
      cardId: json["card_id"],
      name: json["name"],
      imageUrl: json["image_url"] ?? "",
    );
  }
}
