import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

class ChartPage extends StatelessWidget {
  final Map<String, dynamic> cardData;
  const ChartPage({super.key, required this.cardData});

  Color get _bg => const Color(0xFF071933);
  Color get _cardBorder => const Color(0xFF183A5A);
  Color get _accentYellow => const Color(0xFFF4C542);
  Color get _accentBlue => const Color(0xFF3FA3FF);
  Color get _mutedLine => Colors.white24;

  @override
  Widget build(BuildContext context) {
    final prices = List<double>.from(cardData['prices'] ?? []);
    final trend = List<double>.from(cardData['trend'] ?? []);
    final last7 = List<double>.from(cardData['last7'] ?? []);
    final last7trend = List<double>.from(cardData['last7trend'] ?? []);

    return Scaffold(
      backgroundColor: _bg,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new),
          onPressed: () => Navigator.pop(context),
        ),
        title: const SizedBox.shrink(),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 18.0, vertical: 8),
          child: Column(
            children: [
              Expanded(
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: const Color(0x0AFFFFFF), // very subtle overlay
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(color: _cardBorder),
                  ),
                  padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      // Title
                      Text(
                        cardData['name'] ?? 'Unknown',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: _accentYellow,
                          fontSize: 36,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 0.8,
                          shadows: [
                            Shadow(color: Colors.black.withOpacity(0.35), offset: const Offset(0, 2), blurRadius: 4),
                          ],
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        cardData['subtitle'] ?? '',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: _accentYellow.withOpacity(0.9),
                          fontSize: 18,
                        ),
                      ),
                      const SizedBox(height: 18),

                      // Price pill row
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(36),
                          border: Border.all(color: _cardBorder),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            _condBox('HP', cardData['hp_price'] ?? '\$0', _accentYellow),
                            _condBox('NM', cardData['nm_price'] ?? '\$0', _accentBlue),
                            _condBox('M', cardData['m_price'] ?? '\$0', _accentBlue),
                          ],
                        ),
                      ),

                      const SizedBox(height: 18),

                      // Big chart (flex)
                      Expanded(
                        child: _buildChartBlock(
                          title: null,
                          prices: prices,
                          trend: trend,
                          yMax: _calcMax(prices, trend),
                        ),
                      ),

                      const SizedBox(height: 8),

                      // Last 7 days label
                      const Padding(
                        padding: EdgeInsets.only(top: 6, bottom: 6),
                        child: Text('Last 7 days',
                            style: TextStyle(fontSize: 18, color: Colors.white70, fontWeight: FontWeight.w600)),
                      ),

                      // small chart block
                      SizedBox(
                        height: 140,
                        child: _buildChartBlock(
                          title: null,
                          prices: last7,
                          trend: last7trend,
                          yMax: _calcMax(last7, last7trend),
                          showGridHorizontalInterval: 20,
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 18),

              // bottom logo
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 54,
                    height: 54,
                    decoration: BoxDecoration(
                      color: const Color(0xFF0F3350),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.credit_card, color: Colors.white, size: 30),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    'CardIQ',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 24,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
            ],
          ),
        ),
      ),
    );
  }

  // helper to build condition box (HP / NM / M)
  Widget _condBox(String label, String value, Color accent) {
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text(label, style: TextStyle(color: accent, fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 6),
          Text(value, style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  double _calcMax(List<double> a, List<double> b) {
    final combo = [...a, ...b];
    if (combo.isEmpty) return 60;
    final mx = combo.reduce((v, e) => v > e ? v : e);
    // round up to nearest 10/20
    final step = 10;
    return ((mx / step).ceil() * step).toDouble().clamp(20.0, 200.0);
  }

  Widget _buildChartBlock({
    required List<double> prices,
    required List<double> trend,
    required double yMax,
    double showGridHorizontalInterval = 20,
    String? title,
  }) {
    final spots = <FlSpot>[];
    for (var i = 0; i < prices.length; i++) {
      spots.add(FlSpot(i.toDouble(), prices[i]));
    }
    final trendSpots = <FlSpot>[];
    for (var i = 0; i < trend.length; i++) {
      trendSpots.add(FlSpot(i.toDouble(), trend[i]));
    }

    final maxX = (prices.isNotEmpty ? prices.length - 1 : 1).toDouble();
    final maxY = yMax;

    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(14),
        // slightly darker panel inside
        color: Colors.transparent,
      ),
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: LineChart(
        LineChartData(
          maxX: maxX,
          minX: 0,
          maxY: maxY,
          minY: 0,
          gridData: FlGridData(
            show: true,
            horizontalInterval: showGridHorizontalInterval,
            getDrawingHorizontalLine: (value) => FlLine(color: _mutedLine, strokeWidth: 1),
            drawVerticalLine: false,
          ),
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                interval: showGridHorizontalInterval,
                reservedSize: 40,
                getTitlesWidget: (value, meta) {
                  // show 0/20/40/60 style labels
                  return Text(value.toInt().toString(), style: const TextStyle(color: Colors.white54, fontSize: 12));
                },
              ),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(showTitles: false),
            ),
            rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
            topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          ),
          borderData: FlBorderData(
            show: false,
          ),
          lineBarsData: [
            // solid blue actual
            LineChartBarData(
              spots: spots,
              isCurved: true,
              color: _accentBlue,
              barWidth: 3,
              isStrokeCapRound: true,
              dotData: FlDotData(show: false),
              belowBarData: BarAreaData(
                show: true,
                color: _accentBlue.withOpacity(0.15),
              ),
            ),

            // dashed yellow trend
            LineChartBarData(
              spots: trendSpots,
              isCurved: true,
              color: _accentYellow,
              barWidth: 2.5,
              isStrokeCapRound: true,
              dotData: FlDotData(show: false),
              belowBarData: BarAreaData(show: false),
              // dash array: many fl_chart versions accept this - if not, remove this line
              dashArray: [6, 4],
            ),
          ],
        ),
      ),
    );
  }
}
