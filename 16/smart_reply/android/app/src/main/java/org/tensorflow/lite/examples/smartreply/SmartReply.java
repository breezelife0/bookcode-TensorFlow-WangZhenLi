package org.tensorflow.lite.examples.smartreply;

import androidx.annotation.Keep;

/**
 * SmartReply包含预测的回复信息
 * *<p>注意：不应该混淆JNI使用的这个类、类名和构造函数.
 */
@Keep
public class SmartReply {

  private final String text;
  private final float score;

  @Keep
  public SmartReply(String text, float score) {
    this.text = text;
    this.score = score;
  }

  public String getText() {
    return text;
  }

  public float getScore() {
    return score;
  }
}
