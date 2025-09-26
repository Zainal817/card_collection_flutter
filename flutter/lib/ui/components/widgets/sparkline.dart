import 'package:flutter/material.dart';

class PriceSparkline extends StatelessWidget {
  final List<double> prices;
  const PriceSparkline({super.key, required this.prices});

  @override
  Widget build(BuildContext context) {
    if (prices.isEmpty) return const SizedBox.shrink();
    final maxPrice = prices.reduce((a, b) => a > b ? a : b);
    return Row(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: prices.map((p) {
        final heightFactor = (p / maxPrice).clamp(0.05, 1.0);
        return Expanded(
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 2),
            height: 100 * heightFactor,
            decoration: BoxDecoration(
              color: Colors.blueAccent,
              borderRadius: BorderRadius.circular(4),
            ),
          ),
        );
      }).toList(),
    );
  }
}
