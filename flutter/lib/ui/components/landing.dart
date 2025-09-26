import 'package:flutter/material.dart';
import './scan_header.dart';

class LandingPage extends StatelessWidget {
  const LandingPage({super.key});

  @override
  Widget build(BuildContext context) {
    final bottomPad = MediaQuery.of(context).padding.bottom;
    // total bottom area height = chips area (84) + divider (1) + nav (70) + bottomPad
    final bottomAreaHeight = 84 + 1 + 70 + bottomPad;

    return Scaffold(
      backgroundColor: const Color(0xFF050505),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(vertical: 12),
          child: Column(
            children: [
              // Top stats row
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
                child: Row(
                  children: [
                    const Icon(Icons.local_fire_department, color: Colors.orange),
                    const SizedBox(width: 6),
                    const Text("47", style: TextStyle(fontWeight: FontWeight.bold)),
                    const Spacer(),
                    const Text("\$2,847", style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(width: 8),
                    Row(
                      children: const [
                        Icon(Icons.trending_up, size: 16, color: Colors.greenAccent),
                        SizedBox(width: 4),
                        Text("+\$23 today", style: TextStyle(color: Colors.greenAccent)),
                      ],
                    ),
                    const Spacer(),
                    const Icon(Icons.notifications_none),
                    const SizedBox(width: 8),
                    const Icon(Icons.settings_outlined),
                  ],
                ),
              ),

              const SizedBox(height: 8),

              // Scan header (center)
              const ScanHeader(onScanTap: null),

              const SizedBox(height: 8),

              // Two cards row + progress card below
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Column(
                  children: const [
                    _TopCardsRow(),
                    SizedBox(height: 12),
                    _ProgressCard(),
                  ],
                ),
              ),

              // filler space so content is above bottom area
              SizedBox(height: 12),

              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch, // make full width
                    children: const [
                    BottomChipsRow(),
                    SizedBox(height: 12),
                    Divider(height: 1, color: Colors.white24),

                    // just padding instead of Expanded
                    Padding(
                        padding: EdgeInsets.only(top: 8.0),
                        child: BottomNavRow(),
                    ),

                    SizedBox(height: 16),
                    RecentActivityCard(),
                    ],
                ),
             ),
            ],
          ),
        ),
      ),

      // Unified bottom area (chips row + divider + nav row) with fixed height
    //   bottomNavigationBar: SafeArea(
    //     child: SizedBox(
    //       height: 172, // 👈 give enough height for chips + nav bar
    //       child: Column(
    //         mainAxisAlignment: MainAxisAlignment.spaceBetween,
    //         children: const [
    //           BottomChipsRow(),
    //           Divider(height: 1, color: Colors.white24),
    //           Expanded(
    //             child: Padding(
    //               padding: const EdgeInsets.only(top: 8.0),
    //               child: BottomNavRow(),
    //             ),
    //           ),
    //           RecentActivityCard()
    //         ],
    //       ),
    //     ),
    //   ),

    );
  }
}

class _TopCardsRow extends StatelessWidget {
  const _TopCardsRow({super.key});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: const [
        Expanded(child: HotOpportunityCard()),
        SizedBox(width: 12),
        Expanded(child: BestPerformerCard()),
      ],
    );
  }
}

class HotOpportunityCard extends StatelessWidget {
  const HotOpportunityCard({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 220,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Title at the top
          const Text(
            "Today's Hot Opportunity",
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 15,
              color: Colors.white,
            ),
          ),

          const SizedBox(height: 5),

          // Row with icon + price + timer
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              // Yellow trending box
              Container(
                width: 48,
                height: 48,
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Color(0xFFFFA000), Color(0xFFFF6D00)],
                  ),
                  borderRadius: BorderRadius.all(Radius.circular(12)),
                ),
                child: const Icon(Icons.trending_up, color: Colors.white),
              ),

              const SizedBox(width: 10),

              // Price + timer
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: const [
                  Text(
                    "Now \$247 \n(was \$189)",
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 12,
                    ),
                  ),
                  Text(
                    "⏱ 23m left",
                    style: TextStyle(
                      color: Colors.greenAccent,
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 10),

          // Tag pill
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [Color(0xFFFFF176), Color(0xFFFFA726)],
              ),
              borderRadius: BorderRadius.circular(16),
            ),
            child: const Text(
              "TRENDING 🔥 +47%",
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                color: Colors.black,
              ),
            ),
          ),

          const Spacer(), // ✅ Works because card has fixed height

          SizedBox(
            width: double.infinity,
            child: OutlinedButton(
              style: OutlinedButton.styleFrom(
                backgroundColor: const Color(0xFF2A2A2A),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                side: const BorderSide(color: Colors.white24),
                padding: const EdgeInsets.symmetric(vertical: 12),
              ),
              onPressed: () {},
              child: const Text(
                "ADD TO WATCHLIST",
                style: TextStyle(color: Colors.white70, fontSize: 13),
              ),
            ),
          ),
        ],
      )

    );
  }
}


class BestPerformerCard extends StatelessWidget {
  const BestPerformerCard({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 220,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          const Text(
            "Your Best Performer",
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 15,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 20),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Color(0xFFFFD54F), Color(0xFFFF8A65)],
                  ),
                  borderRadius: BorderRadius.all(Radius.circular(12)),
                ),
                child: const Icon(Icons.bolt, color: Colors.white),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: const [
                    Text(
                      "Your Pikachu \nup \$12 \ntoday!",
                      style: TextStyle(
                        color: Colors.greenAccent,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const Spacer(),

          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.pinkAccent,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                padding: const EdgeInsets.symmetric(vertical: 12),
              ),
              onPressed: () {},
              child: const Text(
                "Tell friends \nabout your win!",
                textAlign: TextAlign.center, 
                style: TextStyle(color: Colors.white, fontSize: 13),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ProgressCard extends StatelessWidget {
  const _ProgressCard({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF2A2A2A), // Card background color
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.3),
            blurRadius: 6,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Left text side
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                "Almost There!",
                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
              ),
              const SizedBox(height: 4),
              Text(
                "Base Set: 45/69 Complete △",
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Colors.white70,
                    ),
              ),
              const SizedBox(height: 4),
              Text(
                "Only 3 rare cards left!",
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Colors.greenAccent,
                      fontStyle: FontStyle.italic,
                    ),
              ),
            ],
          ),

          // Right ? ? ? blocks
          Row(
            children: List.generate(
              3,
              (index) => Container(
                margin: const EdgeInsets.only(left: 6),
                width: 40,
                height: 50,
                decoration: BoxDecoration(
                  color: Colors.grey[800],
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Center(
                  child: Text(
                    "?",
                    style: TextStyle(
                      fontSize: 20,
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/* --------------------------- Chips Row (circle chips with labels below) --------------------------- */
class BottomChipsRow extends StatelessWidget {
  const BottomChipsRow({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: const [
          CircleChip(value: "8.0", label: "Daily Goal", colors: [Color(0xFF8A2BE2), Color(0xFF4169E1)]),
          CircleChip(value: "47 \n days", label: "Streak", colors: [Color(0xFF00C853), Color(0xFF69F0AE)]),
          CircleChip(value: "Level \n 37/100", label: "Level", colors: [Color(0xFFFF4081), Color(0xFFF50057)]),
          CircleChip(value: "Local", label: "Rank", colors: [Color(0xFFFF9800), Color(0xFFFFD180)]),
        ],
      ),
    );
  }
}

class CircleChip extends StatelessWidget {
  final String value;
  final String label;
  final List<Color> colors;
  const CircleChip({super.key, required this.value, required this.label, required this.colors});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          width: 68,
          height: 68,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            gradient: LinearGradient(colors: colors, begin: Alignment.topLeft, end: Alignment.bottomRight),
            boxShadow: [BoxShadow(color: colors.last.withOpacity(0.35), blurRadius: 8, offset: const Offset(0,4))],
          ),
          child: Center(child: Text(value, textAlign: TextAlign.center, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold))),
        ),
        const SizedBox(height: 8),
        Text(label, style: const TextStyle(color: Colors.white70, fontSize: 12)),
      ],
    );
  }
}

/* --------------------------- Bottom navigation row (rectangular buttons with icons + labels) --------------------------- */
class BottomNavRow extends StatelessWidget {
  const BottomNavRow({super.key});

  Widget _navButton({required BuildContext context, required IconData icon, required String label, int badge = 0, String? route}) {
    return Expanded(
        child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 2),
        child: InkWell(
            onTap: () {
                if (route != null) Navigator.pushNamed(context, route);
            },
            borderRadius: BorderRadius.circular(14),
            child: Container(
            padding: const EdgeInsets.symmetric(vertical: 10),
            decoration: BoxDecoration(
                color: const Color(0xFF1E1E1E),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: Colors.white10),
            ),
            child: Stack(
                children: [
                Align(
                    alignment: Alignment.center,
                    child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                        Icon(icon, color: Colors.white70, size: 22),
                        const SizedBox(height: 6),
                        Text(
                        label,
                        style: const TextStyle(
                            fontSize: 11,
                            color: Colors.white70,
                        ),
                        ),
                    ],
                    ),
                ),
                if (badge > 0)
                  Stack(
                    clipBehavior: Clip.none,
                    children: [
                      // Badge on top (z-index higher because it's later in the list)
                      Positioned(
                        top: -15,
                        right: 0,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: Colors.red,
                            borderRadius: BorderRadius.circular(12),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.3),
                                blurRadius: 6,
                                offset: const Offset(2, 3),
                              ),
                            ],
                          ),
                          child: Text(
                            "$badge",
                            style: TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                            ),
                          ),
                        ),
                      ),
                    ],
                  )
                ],
              ),
            ),
          ),
        ),
      );
    }

    @override
    Widget build(BuildContext context) {
        return SizedBox(
        height: 70,
        child: Row(
            children: [
            _navButton(context: context, icon: Icons.qr_code_scanner, label: "SCAN CARD", route : '/scan'),
            _navButton(context: context, icon: Icons.collections, label: "COLLECTION", badge: 4, route : '/collection'),
            _navButton(context: context, icon: Icons.swap_horiz, label: "TRADE"),
            _navButton(context: context, icon: Icons.people, label: "FRIENDS"),
            ],
        ),
        );
    }
    }

class RecentActivityCard extends StatelessWidget {
  const RecentActivityCard({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Title
          const Text(
            "Recent Activity",
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 12),

          // Activity items
          const _ActivityItem(
            icon: Icons.bolt, // ⚡ substitute for event
            message: "Mike found a \$89 Blastoise 2.1 miles away!",
            time: "2m ago",
          ),
          const SizedBox(height: 10),

          const _ActivityItem(
            icon: Icons.star,
            message: "You completed Gym Heroes set!",
            time: "5m ago",
          ),
          const SizedBox(height: 10),

          const _ActivityItem(
            icon: Icons.trending_up,
            message: "Your Charizard gained \$4 in the last hour",
            time: "1h ago",
          ),
        ],
      ),
    );
  }
}

class _ActivityItem extends StatelessWidget {
  final IconData icon;
  final String message;
  final String time;

  const _ActivityItem({
    required this.icon,
    required this.message,
    required this.time,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Icon bubble
        Container(
          width: 28,
          height: 28,
          decoration: BoxDecoration(
            color: Colors.white10,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, size: 16, color: Colors.white70),
        ),
        const SizedBox(width: 12),

        // Text content
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                message,
                style: const TextStyle(color: Colors.white, fontSize: 13),
              ),
              const SizedBox(height: 4),
              Text(
                time,
                style: const TextStyle(color: Colors.white54, fontSize: 11),
              ),
            ],
          ),
        ),
      ],
    );
  }
}