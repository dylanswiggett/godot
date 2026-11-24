/**************************************************************************/
/*  lightmap_portal.cpp                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/


#include "lightmap_portal.h"

#include "core/object/class_db.h"

void LightmapPortal::set_linked_portal(const NodePath &p_path) {
	linked_portal = p_path;
	update_configuration_warnings();
}

NodePath LightmapPortal::get_linked_portal() const {
	return linked_portal;
}

LightmapPortal *LightmapPortal::get_linked_portal_node() const {
	if (!is_inside_tree() || linked_portal.is_empty()) {
		return nullptr;
	}
	LightmapPortal *linked = Object::cast_to<LightmapPortal>(get_node_or_null(linked_portal));
	return linked != this ? linked : nullptr;
}

PackedStringArray LightmapPortal::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();
	if (is_inside_tree() && !get_linked_portal_node()) {
		warnings.push_back(RTR("Linked Portal must point to another LightmapPortal, or this portal is ignored when baking."));
	}
	return warnings;
}

void LightmapPortal::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_linked_portal", "path"), &LightmapPortal::set_linked_portal);
	ClassDB::bind_method(D_METHOD("get_linked_portal"), &LightmapPortal::get_linked_portal);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "linked_portal", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "LightmapPortal"), "set_linked_portal", "get_linked_portal");
}

LightmapPortal::LightmapPortal() {
}
