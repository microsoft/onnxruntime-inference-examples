/**
 * Copyright 2019 The TensorFlow Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Portions Copyright (c) Microsoft Corporation

import UIKit

// This file creates an overlay view to be drawn on a detected object.

struct ObjectOverlay {
    let name: String
    let borderRect: CGRect
    let nameStringSize: CGSize
    let color: UIColor
    let font: UIFont
}

class OverlayView: UIView {
    var objectOverlays: [ObjectOverlay] = []
    private let cornerRadius: CGFloat = 10.0
    private let stringBgAlpha: CGFloat = 0.7
    private let lineWidth: CGFloat = 3
    private let stringFontColor = UIColor.white
    private let stringHorizontalSpacing: CGFloat = 13.0
    private let stringVerticalSpacing: CGFloat = 7.0
    
    override func draw(_ rect: CGRect) {
        for objectOverlay in objectOverlays {
            drawBorders(of: objectOverlay)
            drawBackground(of: objectOverlay)
            drawName(of: objectOverlay)
        }
    }
    
    func drawBorders(of objectOverlay: ObjectOverlay) {
        let path = UIBezierPath(rect: objectOverlay.borderRect)
        path.lineWidth = lineWidth
        objectOverlay.color.setStroke()
        
        path.stroke()
    }
    
    func drawBackground(of objectOverlay: ObjectOverlay) {
        let stringBgRect = CGRect(x: objectOverlay.borderRect.origin.x, y: objectOverlay.borderRect.origin.y,
                                  width: 2 * stringHorizontalSpacing + objectOverlay.nameStringSize.width,
                                  height: 2 * stringVerticalSpacing + objectOverlay.nameStringSize.height)
        
        let stringBgPath = UIBezierPath(rect: stringBgRect)
        objectOverlay.color.withAlphaComponent(stringBgAlpha).setFill()
        stringBgPath.fill()
    }
    
    func drawName(of objectOverlay: ObjectOverlay) {
        let stringRect = CGRect(x: objectOverlay.borderRect.origin.x + stringHorizontalSpacing,
                                y: objectOverlay.borderRect.origin.y + stringVerticalSpacing,
                                width: objectOverlay.nameStringSize.width,
                                height: objectOverlay.nameStringSize.height)
        
        let attributedString = NSAttributedString(string: objectOverlay.name,
                                                  attributes: [
                                                      NSAttributedString.Key.foregroundColor: stringFontColor,
                                                      NSAttributedString.Key.font: objectOverlay.font,
                                                  ])
        attributedString.draw(in: stringRect)
    }
}
