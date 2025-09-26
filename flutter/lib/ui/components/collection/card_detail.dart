import 'package:flutter/material.dart';
import '../widgets/chart_page.dart';
import '../screen/price.dart';

class CardDetailScreen extends StatelessWidget {
  final Map<String, dynamic> cardData;

  const CardDetailScreen({super.key, required this.cardData});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1B263B),
      appBar: AppBar(
        title: const Text(
          "CardIQ",
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        centerTitle: true,
        backgroundColor: const Color(0xFF1B263B), // dark appbar
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Card image + name + set info
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: Image.network(
                    cardData['image_url'] ??
                        'https://via.placeholder.com/140x180.png?text=Card',
                    width: 150,
                    height: 190,
                    fit: BoxFit.cover,
                    errorBuilder: (context, error, stack) => Container(
                      width: 150,
                      height: 190,
                      color: Colors.grey.shade800,
                      alignment: Alignment.center,
                      child: const Icon(Icons.broken_image,
                          color: Colors.white54),
                    ),
                  ),
                ),
                const SizedBox(width: 15),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        cardData['name'] ?? "Unknown Card",
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 28,
                          color: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 5),
                      Text(
                        "${cardData['number'] ?? '28/180'}",
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: Colors.white70,
                        ),
                      ),
                      const SizedBox(height: 13),
                      Text(
                        "${cardData['rarity'] ?? 'Uncommon'}\n${cardData['set'] ?? 'Base Set 2'}",
                        style: const TextStyle(
                          color: Colors.white70,
                          fontSize: 16,
                        ),
                      ),
                      const SizedBox(height: 8),
                      InkWell(
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => ChartPage(cardData: {
                                'name': cardData['name'] ?? 'Charizard',
                                'subtitle': cardData['set'] ?? 'Vivid Voltage — 25/185',
                                'hp_price': '\$5.00',
                                'nm_price': '\$7.80',
                                'm_price': '\$52.47',
                                'prices': [20.0, 25.0, 22.0, 28.0, 35.0, 30.0, 34.0],
                                'trend': [15.0, 17.0, 20.0, 25.0, 30.0, 40.0, 50.0],
                                'last7': [20.0, 22.0, 28.0, 27.0, 30.0, 35.0, 40.0],
                                'last7trend': [18.0, 20.0, 23.0, 26.0, 32.0, 38.0, 45.0],
                              }),
                            ),
                          );
                        },
                        child: Container(
                          padding:
                              const EdgeInsets.symmetric(vertical: 12, horizontal: 12),
                          decoration: BoxDecoration(
                            color: const Color(0xFF111827),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Text(
                            "View Price Chart",
                            style: TextStyle(
                              color: Colors.amber,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),

            const SizedBox(height: 20),

            // Price Table
            _priceRow("Mint", cardData['mint_price'] ?? "\$43.00"),
            const Divider(color: Colors.white24, thickness: 0.7),
            _priceRow("Near Mint", cardData['near_mint_price'] ?? "\$15.00"),
            const Divider(color: Colors.white24, thickness: 0.7),
            _priceRow("Lightly Played",
                cardData['lightly_played_price'] ?? "\$8.50"),
            const Divider(color: Colors.white24, thickness: 0.7),
            _priceRow("Heavily Played",
                cardData['heavily_played_price'] ?? "\$2.20"),
            const Divider(color: Colors.white24, thickness: 0.7),

            const SizedBox(height: 12),

            // Recommended Buy Price
            InkWell(
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => PricingAlertsPage(),
                  ),
                );
              },
              borderRadius: BorderRadius.circular(8),
              child: Container(
                padding:
                    const EdgeInsets.symmetric(vertical: 12, horizontal: 12),
                decoration: BoxDecoration(
                  color: const Color(0xFF111827),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      "Recommended Buy Price",
                      style: TextStyle(
                        color: Colors.amber,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    Text(
                      cardData['recommended_price'] ?? "\$14.00",
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    )
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _priceRow(String label, String price) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w500,
              color: Colors.white70,
            ),
          ),
          Text(
            price,
            style: const TextStyle(
              fontSize: 15,
              color: Colors.white,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}
