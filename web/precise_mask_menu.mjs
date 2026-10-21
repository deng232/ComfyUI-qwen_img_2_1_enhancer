export function installPreciseMaskMenu(nodeType, openEditor) {
  const original = nodeType.prototype.getExtraMenuOptions;
  nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
    const returned = original?.call(this, canvas, options);
    if (Array.isArray(returned) && returned !== options && returned.length) options = returned;
    if (!options.some((option) => option?.content === "Open Precise Mask Editor")) {
      options.push({ content: "Open Precise Mask Editor", callback: () => openEditor(this) });
    }
    return options;
  };
}
