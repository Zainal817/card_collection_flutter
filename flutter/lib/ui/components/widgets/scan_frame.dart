import 'package:flutter/material.dart';

class ScanFramePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.blueAccent
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    const inset = 28.0;
    const len = 36.0;

    // corners
    canvas.drawLine(const Offset(inset, inset), const Offset(inset + len, inset), paint);
    canvas.drawLine(const Offset(inset, inset), const Offset(inset, inset + len), paint);

    canvas.drawLine(Offset(size.width - inset, inset),
        Offset(size.width - inset - len, inset), paint);
    canvas.drawLine(Offset(size.width - inset, inset),
        Offset(size.width - inset, inset + len), paint);

    canvas.drawLine(Offset(inset, size.height - inset * 3),
        Offset(inset + len, size.height - inset * 3), paint);
    canvas.drawLine(Offset(inset, size.height - inset * 3),
        Offset(inset, size.height - inset * 3 - len), paint);

    canvas.drawLine(Offset(size.width - inset, size.height - inset * 3),
        Offset(size.width - inset - len, size.height - inset * 3), paint);
    canvas.drawLine(Offset(size.width - inset, size.height - inset * 3),
        Offset(size.width - inset, size.height - inset * 3- len), paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
