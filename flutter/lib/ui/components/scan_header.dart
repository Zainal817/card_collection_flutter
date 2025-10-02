import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_arc_text/flutter_arc_text.dart';

class ScanHeader extends StatefulWidget {
  final VoidCallback? onScanTap;
  final int ringCount;
  final Duration cycleDuration;

  const ScanHeader({
    Key? key,
    this.onScanTap,
    this.ringCount = 6,
    this.cycleDuration = const Duration(seconds: 5),
  }) : super(key: key);

  @override
  State<ScanHeader> createState() => _ScanHeaderState();
}

class _ScanHeaderState extends State<ScanHeader> with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  static const double coreSize = 140.0;
  static const double coreDot = 18.0;
  static const double ringMin = 28.0;
  static const double ringMax = 220.0;
  static const Color ringColor = Color(0xFF4A90E2); // bluish ring
  static const Color coreColorStart = Color(0xFF8A2BE2);
  static const Color coreColorMid = Color(0xFF4169E1);
  static const Color coreColorEnd = Color(0xFF00BFFF);

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: widget.cycleDuration,
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  // progress in [0,1)
  double _staggeredProgress(int index, int total) {
    final double offset = (index / total);
    // shift and wrap in 0..1
    final raw = (_controller.value + offset) % 1.0;
    return raw;
  }

  @override
  Widget build(BuildContext context) {
    final rings = widget.ringCount;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Transform.translate(
          offset: const Offset(0, 24),
          child: ArcText(
            radius: coreSize * 0.95,
            startAngle: -0.45,
            text: "TAP TO SCAN",
            textStyle:
                Theme.of(context).textTheme.labelLarge?.copyWith(
                  letterSpacing: 2,
                  fontWeight: FontWeight.w600,
                  color: Colors.black87.withOpacity(0.85),
                ) ??
                const TextStyle(
                  letterSpacing: 2,
                  fontWeight: FontWeight.w600,
                  color: Colors.black87,
                ),
          ),
        ),

        const SizedBox(height: 12),

        // Tappable scanner area with rings and core
        GestureDetector(
          onTap: widget.onScanTap,
          child: SizedBox(
            width: ringMax,
            height: ringMax,
            child: Stack(
              alignment: Alignment.center,
              children: [
                for (var i = 0; i < rings; i++)
                  AnimatedBuilder(
                    animation: _controller,
                    builder: (context, child) {
                      final p = _staggeredProgress(i, rings); // 0..1
                      final eased = Curves.easeOut.transform(p);
                      final size = ringMin + eased * (ringMax - ringMin);
                      final opacity = (1.0 - eased).clamp(0.0, 1.0);
                      // subtle scale jitter to avoid perfect linear cycle
                      final jitter =
                          1.0 +
                          0.02 * sin((_controller.value + i * 0.35) * 2 * pi);
                      return SizedBox(
                        width: size * jitter,
                        height: size * jitter,
                        child: DecoratedBox(
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: ringColor.withOpacity(0.28 * opacity),
                              width: 1.6,
                            ),
                          ),
                        ),
                      );
                    },
                  ),

                for (var d in [0.72, 1.08, 1.36])
                  SizedBox(
                    width: coreSize * d,
                    height: coreSize * d,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.black12, width: 0.9),
                      ),
                    ),
                  ),

                AnimatedBuilder(
                  animation: _controller,
                  builder: (context, child) {
                    // subtle breathing animation using sin
                    final t = _controller.value;
                    final pulse = 1.0 + 0.035 * sin(t * 2 * pi * 1.5);
                    final glowAlpha =
                        0.35 + 0.15 * (0.5 + 0.5 * sin(t * 2 * pi));
                    return Transform.scale(
                      scale: pulse,
                      child: Container(
                        width: coreSize,
                        height: coreSize,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: RadialGradient(
                            colors: [
                              coreColorStart.withOpacity(0.98),
                              coreColorMid.withOpacity(0.95),
                              coreColorEnd.withOpacity(0.9),
                            ],
                            stops: const [0.18, 0.6, 1.0],
                            radius: 0.9,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: coreColorMid.withOpacity(glowAlpha),
                              blurRadius: 48,
                              spreadRadius: 16,
                            ),
                          ],
                        ),
                        child: Center(
                          child: Container(
                            width: coreDot,
                            height: coreDot,
                            decoration: const BoxDecoration(
                              shape: BoxShape.circle,
                              color: Colors.white,
                            ),
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
        ),

        const SizedBox(height: 12),

        // Scan stats text
        Text(
          '8/10 daily scans',
          style: Theme.of(
            context,
          ).textTheme.bodySmall?.copyWith(color: Colors.white70),
        ),
        const SizedBox(height: 4),
        Text(
          '2 scans until BONUS ROUND!',
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
            color: Colors.pinkAccent,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 12),
      ],
    );
  }
}
